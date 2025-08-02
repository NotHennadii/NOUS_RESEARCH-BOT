import requests
import random
import time
import threading
import asyncio
import aiohttp
from itertools import cycle
from rich.text import Text
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
from pathlib import Path

# Initialize Rich Console with enhanced features
console = Console(record=True, width=120)

# Configuration
API_URL = "https://inference-api.nousresearch.com/v1/chat/completions"

MODEL_WEIGHTS = {
    "Hermes-3-Llama-3.1-405B": 5,
    "Hermes-3-Llama-3.1-70B": 15,
    "DeepHermes-3-Mistral-24B-Preview": 30,
    "DeepHermes-3-Llama-3-8B-Preview": 50
}

MAX_RETRIES = 3
MIN_QUESTION_LENGTH = 30  # Уменьшил для экономии токенов
MAX_QUESTION_LENGTH = 40  # Уменьшил с 60 до 50 для экономии токенов

# Global statistics
@dataclass
class Stats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    start_time: Optional[datetime] = None
    avg_response_time: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

stats = Stats()
stats_lock = threading.Lock()

def print_modern_banner():
    """Original banner with token optimization notice"""
    banner_text = Text(
        """
*--------------------------------------------------------------------------*..=#####**=.  
*                            ...:-==++++==-:..                             *..=##=::+##=. 
*                       .:-:.....::-+*######***+-.                         *..=##*++*#*-. 
*                   .-+*********+:. ...=*########**=.                      *..=##*++=-..  
*                .=*###############*:   .-###########+.                    *..=##-.       
*            ..-*####################*:.  .+###########+..                 *..=**-.       
*          ..=#####################****=.. .-*#######**#:-.                *.   .....     
*        ..=*#####**###########++*+. .-++.  .-*##*##*+:=.==.               *...-**##*=:.  
*       .:=+*####++*##*+***###*+:-...-+*==   .==+:=+=.. -**=.              *..+#*=::*#*-. 
*      .:-:::+*++-:-==:::  --:.. .=**###**+   .+*==***++**#*-.             *.-##+.  -##+. 
*      :=:...::======++****+*+-==+*#######*:   :##**###**###*:             *.-##+.  -##+. 
*     .*+******#*##########################+.-=+*############*.            *..*#*-..*#*-. 
*     =#################**#################*-+#=*###**########-            *...=*####*:.  
*    .*#################*+*################***##++*##+*########.           *........      
*    :*##################=+*##########################*########-           *..+******+-.. 
*    :*##########*+*#####*-+##########################**#######*.          *..+##=--*##=. 
*    .*##########*=:=+=++=-::=======+**################=#######*:          *..+##===*#*-. 
*     :*######*==:   ..:=*#**+-..    .=################**######*=.         *..+##+*##+:.  
*      .+#***#*=+=:.   .--*##+-++..  .=######+===-+#####+*######+.         *..+##:.*#*=.  
*      ...=+-+#*=#+.   ..::=+-=-..   .=#################*+######*-         *..+##:.:##*:. 
*          ..:#*==:.    ........     :+##################**######=.        *....... ..... 
*            -#*:                   .-###################*+*#####*:        *..+********:  
*            +*+-.:..               .-####################**######=.       *..=++*##*++:  
*           .**-.::                 :+####################***#####*:.      *.   .+##-     
*           .**=.-:..              .:######################**######-.      *.   .+##-     
*           :*#*:-*++=-:           .-######*###############**##**##*.      *.   .+##-     
*           -*#*+:--....           .*#####*-###################**+##=      *.   .+##-     
*    ..    .+###*+:..              :######-:####################+-*#*.     *.   ..::.     
*  .-:.    .+#####+.               :#####*.:*###################+:=##-     *.   :+++-     
*  -+:     -*######+.              =#####+######################+.=*#*.::  *.  .+###*:    
* .=*:.   .+#########*+***###*+:.  +####**-.-###################+.=*##.:+-.*. .-**+##+.   
* .=**=:::+###**###############*-. -####*...=##################*-:*###..+=.*. :+#=.*#*-.  
* :=+*######*+-##################==#*#####*####################*:+####.:++.*..=#######+:  
* .=+=+*#*+=:=###**###############+::*#########################=*####*.-*+.*.:**=:::+##=. 
*  .=*+=---=*###*+*############*+:. ..=*#############################::**-.*.:-:.   .--:. 
*   ..=*#######+-+###########*=..     ..:+#############*############--**=. *. .-==:       
*      ..:+=::.:*#############*=.     .:+**++-=*#+-::::-=+--*#####*-+#*=.  *. .=##-       
*        .-****###++*###########=. .-**+:.:.=**:.       ..:+==###***#=..   *. .=##-       
*          .:=++-.:*###########*#***+:..-.:**-.             -*-*#*=..      *. .=##-       
*               .:+##########*+*+..  .::.+*+.                .*=+.         *. .=##-       
*...............-*#####**+=**+*-....:+=.:*-....................=:..........*. .=##-....   
----------------------------------------------------------------------------. .:------:.          

                         [NOUS RESEARCH FUCKER BOT v3.0]
                  Built for the bleak cold vacuum of rate limits 
""",
        style="bold cyan"
    )
    console.print(banner_text)

def choose_model():
    """Smart model selection with weighted random choice"""
    models = list(MODEL_WEIGHTS.keys())
    weights = list(MODEL_WEIGHTS.values())
    return random.choices(models, weights=weights, k=1)[0]

def load_config_file(file_path: str) -> List[str]:
    """Enhanced file loading with better error handling"""
    try:
        path = Path(file_path)
        if not path.exists():
            console.print(f"⚠️  [yellow]File not found: {file_path}[/yellow]")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        console.print(f"✅ [green]Loaded {len(lines)} entries from {file_path}[/green]")
        return lines
    except Exception as e:
        console.print(f"❌ [red]Error loading {file_path}: {e}[/red]")
        return []

def load_prompts_for_profiles(file_path: str, num_profiles: int) -> List[str]:
    """Загрузка промптов для каждого профиля отдельно"""
    prompts = load_config_file(file_path)
    
    if not prompts:
        # Дефолтный промпт если файл не найден - СДЕЛАНО БОЛЕЕ КОНСАЙЗНЫМ
        default_prompt = "Ты краткий AI. Отвечай очень коротко, буквально в пару слов " 
        return [default_prompt] * num_profiles
    
    # Если промптов меньше чем профилей, циклически повторяем
    if len(prompts) < num_profiles:
        extended_prompts = []
        for i in range(num_profiles):
            extended_prompts.append(prompts[i % len(prompts)])
        return extended_prompts
    
    # Если промптов больше или равно, берем первые num_profiles
    return prompts[:num_profiles]

def estimate_tokens(text: str) -> int:
    """Более точная оценка токенов"""
    # Приблизительная формула: 1 токен ≈ 4 символа для латиницы, 2-3 для кириллицы
    if any(ord(char) > 127 for char in text):  # Есть не-ASCII символы
        return max(1, len(text) // 2)  # Для кириллицы
    else:
        return max(1, len(text) // 4)  # Для латиницы

async def send_async_request(session: aiohttp.ClientSession, prompt: str, api_key: str, 
                           system_prompt: str, max_tokens: int, temperature: float = 0.7, 
                           proxy: Optional[str] = None) -> dict:
    """Async request sender with improved token control"""
    model_name = choose_model()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Оценка входящих токенов
    input_tokens = estimate_tokens(system_prompt + prompt)
    
    # Проверяем, что входящие токены не превышают разумный лимит
    if input_tokens > 1000:  # Если промпт слишком длинный
        # Обрезаем системный промпт
        max_system_length = 200
        if len(system_prompt) > max_system_length:
            system_prompt = system_prompt[:max_system_length] + "..."
            console.print(f"⚠️  [yellow]System prompt truncated to save tokens[/yellow]")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    proxy_url = proxy if proxy else None
    
    for attempt in range(MAX_RETRIES):
        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(API_URL, headers=headers, json=payload, 
                                  proxy=proxy_url, timeout=timeout) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    
                    # Подсчет реальных токенов
                    output_tokens = estimate_tokens(content)
                    total_tokens = input_tokens + output_tokens
                    
                    # Update stats
                    with stats_lock:
                        stats.successful_requests += 1
                        stats.total_requests += 1
                        stats.total_tokens += total_tokens
                        stats.total_input_tokens += input_tokens
                        stats.total_output_tokens += output_tokens
                        if stats.successful_requests > 0:
                            stats.avg_response_time = (stats.avg_response_time * (stats.successful_requests - 1) + elapsed) / stats.successful_requests
                    
                    return {
                        "success": True,
                        "content": content,
                        "model": model_name,
                        "elapsed": elapsed,
                        "prompt": prompt,
                        "total_tokens": total_tokens,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    }
                
                elif response.status == 429:
                    console.print(f"⏳ [yellow]Rate limit hit, waiting {10 * (attempt + 1)}s...[/yellow]")
                    await asyncio.sleep(10 * (attempt + 1))
                    continue
                else:
                    error_text = await response.text()
                    with stats_lock:
                        stats.failed_requests += 1
                        stats.total_requests += 1
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "elapsed": elapsed
                    }
                    
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                with stats_lock:
                    stats.failed_requests += 1
                    stats.total_requests += 1
                return {
                    "success": False,
                    "error": str(e),
                    "elapsed": time.time() - start_time
                }
            await asyncio.sleep(5 * (attempt + 1))

async def generate_question_async(session: aiohttp.ClientSession, api_key: str, 
                                system_prompt: str, max_tokens: int, temperature: float,
                                lang: str = 'r', proxy: Optional[str] = None) -> Optional[str]:
    """Async question generator with optimized prompts"""
    # Более короткие промпты для экономии токенов
    prompts = {
        'r': f"Короткий вопрос о технологиях/ИИ ({MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} символов):",
        'e': f"Short tech/AI question ({MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} chars):",
        'b': random.choice([
            f"Короткий вопрос о технологиях/ИИ ({MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} символов):",
            f"Short tech/AI question ({MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} chars):"
        ])
    }
    
    # Используем меньший лимит токенов для генерации вопросов
    question_max_tokens = min(max_tokens // 2, 32)
    
    result = await send_async_request(
        session, prompts.get(lang, prompts['r']), 
        api_key, system_prompt, question_max_tokens, temperature, proxy
    )
    
    if result["success"]:
        question = result["content"].split('\n')[0].strip()
        # Обрезаем до нужной длины
        if len(question) > MAX_QUESTION_LENGTH:
            question = question[:MAX_QUESTION_LENGTH]
        elif len(question) < MIN_QUESTION_LENGTH:
            # Если вопрос слишком короткий, дополняем
            question = question + "?"
        return question
    return None

def format_proxies(raw_proxies: List[str]) -> List[str]:
    """Enhanced proxy formatter"""
    formatted = []
    for line in raw_proxies:
        line = line.strip()
        if not line:
            continue
            
        if "@" not in line and line.count(":") == 3:
            try:
                login, password, ip, port = line.split(":")
                formatted.append(f"http://{login}:{password}@{ip}:{port}")
            except ValueError:
                console.print(f"⚠️  [yellow]Invalid proxy format: {line}[/yellow]")
        else:
            formatted.append(line if line.startswith(("http://", "https://", "socks5://")) else f"http://{line}")
    
    return formatted


async def worker_async(worker_id: int, count: int, delay_range: Tuple[float, float], 
                      api_keys: cycle, proxy: Optional[str], system_prompt: str, 
                      lang: str, max_tokens: int, temperature: float, progress: Progress, task_id: int):
    """Enhanced async worker with precise token control"""
    
    async with aiohttp.ClientSession() as session:
        for i in range(count):
            try:
                api_key = next(api_keys)
                
                # Generate question
                question = await generate_question_async(
                    session, api_key, system_prompt, max_tokens, temperature, lang, proxy
                )
                
                if question:
                    # Send main request with remaining tokens
                    remaining_tokens = max_tokens - 10  # Резерв для безопасности
                    result = await send_async_request(
                        session, question, api_key, system_prompt, remaining_tokens, temperature, proxy
                    )
                    
                    if result["success"]:
                        # Log success with detailed token info
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        console.print(f"\n[{timestamp}] ✅ [green]Worker-{worker_id}[/green] | [blue]{result['model']}[/blue] | {result['elapsed']:.2f}s")
                        console.print(f"🔤 [cyan]Tokens: {result['total_tokens']} (in:{result['input_tokens']} + out:{result['output_tokens']})[/cyan]")
                        console.print(f"[yellow]Q:[/yellow] {question}")
                        console.print(f"[green]A:[/green] {result['content'][:150]}{'...' if len(result['content']) > 150 else ''}")
                        
                        # Save to log with detailed token tracking
                        try:
                            with open("ai_stress_log.txt", "a", encoding="utf-8") as f:
                                f.write(f"\n[{datetime.now().isoformat()}] Worker-{worker_id} | {result['model']} | {result['elapsed']:.2f}s\n")
                                f.write(f"Tokens: {result['total_tokens']} (Input: {result['input_tokens']}, Output: {result['output_tokens']})\n")
                                f.write(f"Temperature: {temperature} | Max tokens: {max_tokens}\n")
                                f.write(f"Question: {question}\n")
                                f.write(f"Answer: {result['content']}\n")
                                f.write("="*80 + "\n")
                        except Exception as log_error:
                            console.print(f"⚠️  [yellow]Log error: {log_error}[/yellow]")
                    else:
                        console.print(f"❌ [red]Worker-{worker_id} failed:[/red] {result['error']}")
                else:
                    console.print(f"⚠️  [yellow]Worker-{worker_id}: Failed to generate question[/yellow]")
                
                progress.update(task_id, advance=1)
                
                if i < count - 1:
                    sleep_time = random.uniform(delay_range[0], delay_range[1])
                    await asyncio.sleep(max(0.1, sleep_time))
                    
            except Exception as worker_error:
                console.print(f"❌ [red]Worker-{worker_id} error: {worker_error}[/red]")
                progress.update(task_id, advance=1)

def get_enhanced_user_inputs():
    """Modern interactive input system with token and temperature control"""
    console.print("\n🎛️[bold bright_cyan] МЕНЮ НАСТРОЕК (TOKEN CONTROL) [/bold bright_cyan]")
    
    try:
        # Requests per thread
        count_input = console.input("[bold]🔢 Количество запросов к AI [dim](default: 5)[/dim]: [/bold]")
        count = int(count_input) if count_input.strip() else 5
        
        # Token limit - strict control - Изменил рекомендуемый диапазон
        tokens_input = console.input("[bold]🔤 Максимальное кол-во токенов [dim](default: 80, рекомендуется 50-80 для экономии)[/dim]: [/bold]")
        max_tokens = int(tokens_input) if tokens_input.strip() else 80
        
        # Temperature control
        temp_input = console.input("[bold]🌡️ Температура (креативность) [dim](0.1-2.0, default: 0.7)[/dim]: [/bold]")
        temperature = float(temp_input) if temp_input.strip() else 0.7
        temperature = max(0.1, min(2.0, temperature))  # Ограничиваем диапазон
        
        # Delay configuration
        delay_input = console.input("[bold]⏱️ Задержка между запросами [dim](e.g., '1 3' for random between 1-3, default: 2)[/dim]: [/bold]")
        if not delay_input.strip():
            delay_input = "2"
        delay_parts = delay_input.strip().split()
        
        if len(delay_parts) == 2:
            delay_min, delay_max = float(delay_parts[0]), float(delay_parts[1])
            if delay_min > delay_max:
                delay_min, delay_max = delay_max, delay_min
        else:
            delay_min = delay_max = float(delay_parts[0])
        
        # Thread count
        threads_input = console.input("[bold]🧵 Количество потоков [dim](default: 1)[/dim]: [/bold]")
        threads = int(threads_input) if threads_input.strip() else 1
        
        # Language selection
        lang_options = {"r": "Russian", "e": "English", "b": "Both"}
        console.print("\n[bold]🌍 Выбери язык:[/bold]")
        for key, value in lang_options.items():
            console.print(f"  [cyan]{key}[/cyan] - {value}")
        
        lang_input = console.input("[bold]Выбери язык [dim](default: b)[/dim]: [/bold]")
        lang_choice = lang_input.strip().lower() if lang_input.strip() else 'b'
        
        # Proxy configuration
        proxy_choices = []
        for i in range(threads):
            proxy_input = console.input(f"[bold]🌐 Использовать прокси к профилю {i+1}? [dim](y/n, default: n)[/dim]: [/bold]")
            use_proxy = proxy_input.strip().lower() == 'y'
            proxy_choices.append(use_proxy)
        
        return count, (delay_min, delay_max), threads, lang_choice, proxy_choices, max_tokens, temperature
        
    except (ValueError, KeyboardInterrupt):
        console.print("❌ [red]Invalid input or interrupted. Please try again.[/red]")
        return get_enhanced_user_inputs()

async def main():
    """Enhanced main function with precise token control"""
    print_modern_banner()
    
    # Get user configuration
    count, delay_range, threads, lang_choice, proxy_choices, max_tokens, temperature = get_enhanced_user_inputs()
    
    # Display configuration
    config_table = Table(title="⚙️ Суммарная настройка", border_style="bright_cyan")
    config_table.add_column("Настройка", style="bright_magenta")
    config_table.add_column("Значение", style="bright_cyan")
    
    config_table.add_row("Кол-во запросов", str(count))
    config_table.add_row("Пауза между запросами", f"{delay_range[0]:.1f}s - {delay_range[1]:.1f}s")
    config_table.add_row("🔤 Лимит токенов", str(max_tokens))
    config_table.add_row("🌡️ Температура", str(temperature))
    config_table.add_row("Кол-во потоков", str(threads))
    config_table.add_row("Язык", {"r": "Russian", "e": "English", "b": "Both"}[lang_choice])
    config_table.add_row("Прокси", str(sum(proxy_choices)))
    
    console.print("\n")
    console.print(config_table)
    
    # Подсчет примерного потребления токенов
    estimated_total_tokens = count * threads * max_tokens * 2  # x2 для вопросов и ответов
    console.print(f"💰 [yellow]Примерное потребление токенов: ~{estimated_total_tokens:,}[/yellow]")
    
    console.input("\n[bold bright_yellow]Жмем Enter для начала...[/bold bright_yellow]")
    
    # Load configuration files
    api_keys = load_config_file("API_keys.txt")
    if not api_keys:
        console.print("❌ [red]No API keys found. Please add keys to API_keys.txt[/red]")
        console.print("💡 [blue]Create API_keys.txt file and add your API keys (one per line)[/blue]")
        return
    
    # Загружаем промпты для каждого профиля
    system_prompts = load_prompts_for_profiles("promt.txt", threads)
    
    raw_proxies = load_config_file("proxy.txt")
    formatted_proxies = format_proxies(raw_proxies)
    proxy_pool = cycle(formatted_proxies) if formatted_proxies else cycle([None])
    
    keys_cycle = cycle(api_keys)
    
    # Initialize statistics
    stats.start_time = datetime.now()
    
    # Create progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        
        # Create tasks for each worker
        tasks = []
        worker_tasks = []
        
        for i in range(threads):
            proxy = next(proxy_pool) if proxy_choices[i] else None
            system_prompt = system_prompts[i]  # Уникальный промпт для каждого профиля
            task_id = progress.add_task(f"Worker-{i+1}", total=count)
            tasks.append(task_id)
            
            worker_task = worker_async(
                i+1, count, delay_range, keys_cycle, proxy, 
                system_prompt, lang_choice, max_tokens, temperature, progress, task_id
            )
            worker_tasks.append(worker_task)
        
        # Run all workers concurrently
        console.print(f"\n🚀 [bold bright_green]Starting {threads} workers with {count} requests each...[/bold bright_green]")
        console.print(f"🔤 [cyan]Token limit per response: {max_tokens} | Temperature: {temperature}[/cyan]\n")
        
        # Create live stats display in separate task
        stats_task = asyncio.create_task(update_stats_display())
        
        try:
            await asyncio.gather(*worker_tasks)
        except KeyboardInterrupt:
            console.print("\n⚠️  [yellow]Interrupted by user[/yellow]")
        finally:
            stats_task.cancel()
    
    # Final summary
    console.print("\n" + "="*80)
    console.print(create_stats_table())
    
    # Token usage summary
    with stats_lock:
        console.print(f"\n💰 [bold bright_green]ИТОГОВАЯ СТАТИСТИКА ТОКЕНОВ:[/bold bright_green]")
        console.print(f"🔤 Всего использовано токенов: [bright_cyan]{stats.total_tokens:,}[/bright_cyan]")
        console.print(f"📥 Входящие токены: [bright_yellow]{stats.total_input_tokens:,}[/bright_yellow]")
        console.print(f"📤 Исходящие токены: [bright_green]{stats.total_output_tokens:,}[/bright_green]")
        
        if stats.successful_requests > 0:
            avg_total = stats.total_tokens / stats.successful_requests
            avg_input = stats.total_input_tokens / stats.successful_requests
            avg_output = stats.total_output_tokens / stats.successful_requests
            console.print(f"📊 Среднее на запрос: [bright_magenta]{avg_total:.1f}[/bright_magenta] токенов ({avg_input:.1f} in + {avg_output:.1f} out)")
    
    console.print("\n✅ [bold bright_green]Stress test completed![/bold bright_green]")
    console.print(f"📄 [cyan]Detailed logs saved to: ai_stress_log.txt[/cyan]")

async def update_stats_display():
    """Update stats display periodically"""
    while True:
        await asyncio.sleep(5)
        # Print current stats every 5 seconds
        console.print("\n" + "─" * 40)
        console.print(create_stats_table())

if __name__ == "__main__":
    try:
        # Check Python version
        if sys.version_info < (3, 7):
            console.print("❌ [red]Python 3.7 or higher is required[/red]")
            sys.exit(1)
            
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 [yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n💥 [red]Unexpected error: {e}[/red]")
        console.print_exception()