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
MIN_RESPONSE_LENGTH = 40
MIN_QUESTION_LENGTH = 30
MAX_QUESTION_LENGTH = 70

# Global statistics
@dataclass
class Stats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    start_time: Optional[datetime] = None
    avg_response_time: float = 0.0

stats = Stats()
stats_lock = threading.Lock()

def print_modern_banner():
    """Original banner with modern enhancements"""
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

async def send_async_request(session: aiohttp.ClientSession, prompt: str, api_key: str, 
                           system_prompt: str, max_tokens: int, proxy: Optional[str] = None) -> dict:
    """Async request sender with improved error handling"""
    model_name = choose_model()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
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
                    
                    # Update stats
                    with stats_lock:
                        stats.successful_requests += 1
                        stats.total_requests += 1
                        stats.total_tokens += len(content.split())
                        if stats.successful_requests > 0:
                            stats.avg_response_time = (stats.avg_response_time * (stats.successful_requests - 1) + elapsed) / stats.successful_requests
                    
                    return {
                        "success": True,
                        "content": content,
                        "model": model_name,
                        "elapsed": elapsed,
                        "prompt": prompt
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
                                system_prompt: str, max_tokens: int, lang: str = 'r', 
                                proxy: Optional[str] = None) -> Optional[str]:
    """Async question generator"""
    prompts = {
        'r': f"Создай уникальный интересный вопрос на русском языке о технологиях, ИИ, криптовалютах или философии. Длина: {MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} символов. Только вопрос, без дополнительного текста.",
        'e': f"Create a unique interesting question in English about technology, AI, cryptocurrency or philosophy. Length: {MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} characters. Just the question, no additional text.",
        'b': random.choice([
            f"Создай уникальный интересный вопрос на русском языке о технологиях, ИИ, криптовалютах или философии. Длина: {MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} символов.",
            f"Create a unique interesting question in English about technology, AI, cryptocurrency or philosophy. Length: {MIN_QUESTION_LENGTH}-{MAX_QUESTION_LENGTH} characters."
        ])
    }
    
    result = await send_async_request(session, prompts.get(lang, prompts['r']), 
                                    api_key, system_prompt, max_tokens, proxy)
    
    if result["success"]:
        question = result["content"].split('\n')[0].strip()
        return question if MIN_QUESTION_LENGTH <= len(question) <= MAX_QUESTION_LENGTH else question[:MAX_QUESTION_LENGTH]
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

def create_stats_table() -> Table:
    """Create a beautiful stats table"""
    table = Table(title="📊 Real-time Statistics", border_style="bright_cyan", title_style="bold bright_white")
    
    table.add_column("Metric", style="bright_magenta", no_wrap=True)
    table.add_column("Value", style="bright_green", justify="right")
    table.add_column("Status", style="bright_cyan", justify="center")
    
    with stats_lock:
        success_rate = (stats.successful_requests / max(stats.total_requests, 1)) * 100
        uptime = str(datetime.now() - stats.start_time).split('.')[0] if stats.start_time else "00:00:00"
        requests_per_min = (stats.total_requests / max((datetime.now() - stats.start_time).total_seconds() / 60, 1)) if stats.start_time else 0
    
    table.add_row("🎯 Total Requests", str(stats.total_requests), "🟢" if stats.total_requests > 0 else "🔴")
    table.add_row("✅ Successful", str(stats.successful_requests), "🟢" if stats.successful_requests > 0 else "🔴")
    table.add_row("❌ Failed", str(stats.failed_requests), "🔴" if stats.failed_requests > 0 else "🟢")
    table.add_row("📈 Success Rate", f"{success_rate:.1f}%", "🟢" if success_rate > 80 else "🟡" if success_rate > 50 else "🔴")
    table.add_row("🔤 Total Tokens", str(stats.total_tokens), "📝")
    table.add_row("⚡ Avg Response", f"{stats.avg_response_time:.2f}s", "🚀" if stats.avg_response_time < 2 else "🐌")
    table.add_row("⏱️  Uptime", uptime, "⏰")
    table.add_row("🔄 Req/Min", f"{requests_per_min:.1f}", "📊")
    
    return table

async def worker_async(worker_id: int, count: int, delay_range: Tuple[float, float], 
                      api_keys: cycle, proxy: Optional[str], system_prompt: str, 
                      lang: str, max_tokens: int, progress: Progress, task_id: int):
    """Enhanced async worker with progress tracking"""
    
    async with aiohttp.ClientSession() as session:
        for i in range(count):
            try:
                api_key = next(api_keys)
                
                # Generate question
                question = await generate_question_async(session, api_key, system_prompt, max_tokens, lang, proxy)
                
                if question:
                    # Send main request
                    result = await send_async_request(session, question, api_key, system_prompt, max_tokens, proxy)
                    
                    if result["success"]:
                        # Log success
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        console.print(f"\n[{timestamp}] ✅ [green]Worker-{worker_id}[/green] | [blue]{result['model']}[/blue] | {result['elapsed']:.2f}s")
                        console.print(f"[yellow]Q:[/yellow] {question}")
                        console.print(f"[green]A:[/green] {result['content'][:100]}{'...' if len(result['content']) > 100 else ''}")
                        
                        # Save to log
                        try:
                            with open("ai_stress_log.txt", "a", encoding="utf-8") as f:
                                f.write(f"\n[{datetime.now().isoformat()}] Worker-{worker_id} | {result['model']} | {result['elapsed']:.2f}s\n")
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
    """Modern interactive input system"""
    console.print("\n🎛️  [bold bright_cyan] МЕНЮ НАСТРОЕК [/bold bright_cyan]")
    
    try:
        # Requests per thread
        count_input = console.input("[bold]🔢 Количество запросов к AI [dim](default: 5)[/dim]: [/bold]")
        count = int(count_input) if count_input.strip() else 5
        
        # Delay configuration
        delay_input = console.input("[bold]⏱️  Задержка между запросами [dim](e.g., '1 3' for random between 1-3, default: 2)[/dim]: [/bold]")
        if not delay_input.strip():
            delay_input = "2"
        delay_parts = delay_input.strip().split()
        
        if len(delay_parts) == 2:
            delay_min, delay_max = float(delay_parts[0]), float(delay_parts[1])
            if delay_min > delay_max:
                delay_min, delay_max = delay_max, delay_min
        else:
            delay_min = delay_max = float(delay_parts[0])
        
        # Token limit
        tokens_input = console.input("[bold]🔤 Максимальное кол-во токенов [dim](default: 128)[/dim]: [/bold]")
        max_tokens = int(tokens_input) if tokens_input.strip() else 128
        
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
        
        return count, (delay_min, delay_max), threads, lang_choice, proxy_choices, max_tokens
        
    except (ValueError, KeyboardInterrupt):
        console.print("❌ [red]Invalid input or interrupted. Please try again.[/red]")
        return get_enhanced_user_inputs()

async def main():
    """Enhanced main function with modern async architecture"""
    print_modern_banner()
    
    # Get user configuration
    count, delay_range, threads, lang_choice, proxy_choices, max_tokens = get_enhanced_user_inputs()
    
    # Display configuration
    config_table = Table(title="⚙️ Суммарная настройка", border_style="bright_cyan")
    config_table.add_column("Настройка", style="bright_magenta")
    config_table.add_column("Значение", style="bright_cyan")
    
    config_table.add_row("Кол-во запросов", str(count))
    config_table.add_row("Пауза между запросами", f"{delay_range[0]:.1f}s - {delay_range[1]:.1f}s")
    config_table.add_row("Кол-во токенов", str(max_tokens))
    config_table.add_row("Кол-во потоков", str(threads))
    config_table.add_row("Язык", {"r": "Russian", "e": "English", "b": "Both"}[lang_choice])
    config_table.add_row("Прокси", str(sum(proxy_choices)))
    
    console.print("\n")
    console.print(config_table)
    
    console.input("\n[bold bright_yellow]Жмем Enter для начала ебки...[/bold bright_yellow]")
    
    # Load configuration files
    api_keys = load_config_file("API_keys.txt")
    if not api_keys:
        console.print("❌ [red]No API keys found. Please add keys to API_keys.txt[/red]")
        console.print("💡 [blue]Create API_keys.txt file and add your API keys (one per line)[/blue]")
        return
    
    system_prompt = "\n".join(load_config_file("promt.txt")) or "You are a helpful AI assistant. Provide concise, informative responses."
    
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
            task_id = progress.add_task(f"Worker-{i+1}", total=count)
            tasks.append(task_id)
            
            worker_task = worker_async(
                i+1, count, delay_range, keys_cycle, proxy, 
                system_prompt, lang_choice, max_tokens, progress, task_id
            )
            worker_tasks.append(worker_task)
        
        # Run all workers concurrently
        console.print(f"\n🚀 [bold bright_green]Starting {threads} workers with {count} requests each...[/bold bright_green]\n")
        
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
    console.print("\n✅ [bold bright_green]Stress test completed![/bold bright_green]")
    console.print(f"📄 [cyan]Detailed logs saved to: ai_stress_log.txt[/cyan]")

async def update_stats_display():
    """Update stats display periodically"""
    while True:
        await asyncio.sleep(2)
        # Stats are updated in real-time by the progress bars

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