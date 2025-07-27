import requests
import random
import time
import threading
from itertools import cycle
from rich.text import Text
from rich.console import Console
from datetime import datetime

console = Console()

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

request_counter = 0
counter_lock = threading.Lock()

def choose_model():
    models = list(MODEL_WEIGHTS.keys())
    weights = list(MODEL_WEIGHTS.values())
    return random.choices(models, weights=weights, k=1)[0]

def print_banner():
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
*               .:+##########*+*+..  .::.+*+.                .*=+.         *. .=##-....   
*...............-*#####**+=**+*-....:+=.:*-....................=:..........*. .=##****+.  
----------------------------------------------------------------------------. .:------:.          

                         [NOUS RESEARCH FUCKER BOT v2.0]
                Built for the bleak cold vacuum of rate limits 
          
""",
        style="bold cyan"
    )
    console.print(banner_text)

def load_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        console.print(f"❌ Файл не найден: {file_path}", style="red")
        return []

def send_prompt(prompt_text, api_key, system_prompt, max_tokens, proxy=None):
    global request_counter
    model_name = choose_model()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None

    retries = 0
    while retries < MAX_RETRIES:
        start_time = datetime.now()
        try:
            response = requests.post(API_URL, headers=headers, json=payload, proxies=proxies, timeout=20)
            elapsed = (datetime.now() - start_time).total_seconds()
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                with counter_lock:
                    request_counter += 1
                    current_count = request_counter
                console.print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ Запрос #{current_count} (модель: {model_name}):\n{prompt_text}\n[green]{content}[/green]\nВремя запроса: {elapsed:.2f} сек\n{'-'*60}\n")
                with open("log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Запрос #{current_count} (модель: {model_name}):\nВопрос: {prompt_text}\nОтвет: {content}\nВремя запроса: {elapsed:.2f} сек\n{'='*80}\n")
                return
            elif response.status_code == 429:
                retries += 1
                console.print(f"⏳ Rate limit на ключ: {api_key} — ждем 10 секунд и повторяем ({retries}/{MAX_RETRIES})...", style="yellow")
                time.sleep(10)
            else:
                console.print(f"❌ Ошибка {response.status_code}: {response.text}\n{'-'*60}", style="red")
                return
        except Exception as e:
            retries += 1
            console.print(f"❌ Ошибка при запросе: {e} ({retries}/{MAX_RETRIES})\n{'-'*60}", style="red")

def generate_question(api_key, system_prompt, max_tokens, proxy=None, lang='r'):
    model_name = choose_model()

    if lang == 'r':
        gen_prompt = f"Придумай уникальный вопрос на русском языке на тему технологий, криптовалют или философии. Длина от {MIN_QUESTION_LENGTH} до {MAX_QUESTION_LENGTH} символов."
    elif lang == 'e':
        gen_prompt = f"Generate a unique question in English about technology, cryptocurrency or philosophy. Length between {MIN_QUESTION_LENGTH} and {MAX_QUESTION_LENGTH} characters."
    else:
        gen_prompt = random.choice([
            f"Придумай уникальный вопрос на русском языке на тему технологий, криптовалют или философии. Длина от {MIN_QUESTION_LENGTH} до {MAX_QUESTION_LENGTH} символов.",
            f"Generate a unique question in English about technology, cryptocurrency or philosophy. Length between {MIN_QUESTION_LENGTH} and {MAX_QUESTION_LENGTH} characters."
        ])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": gen_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens
    }
    proxies = {"http": proxy, "https": proxy} if proxy else None

    retries = 0
    while retries < MAX_RETRIES:
        start_time = datetime.now()
        try:
            response = requests.post(API_URL, headers=headers, json=payload, proxies=proxies, timeout=20)
            elapsed = (datetime.now() - start_time).total_seconds()
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                question = content.split('\n')[0]
                console.print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Сгенерирован вопрос за {elapsed:.2f} сек (модель: {model_name})", style="blue")
                return question
            elif response.status_code == 429:
                retries += 1
                console.print(f"⏳ Rate limit при генерации вопроса. Ждем 10 секунд... ({retries}/{MAX_RETRIES})", style="yellow")
                time.sleep(10)
            else:
                console.print(f"❌ Ошибка генерации вопроса {response.status_code}: {response.text}", style="red")
                return None
        except Exception as e:
            retries += 1
            console.print(f"❌ Ошибка при генерации вопроса: {e} ({retries}/{MAX_RETRIES})", style="red")
    return None

def format_proxies(raw_proxies):
    formatted = []
    for line in raw_proxies:
        if "@" not in line and line.count(":") == 3:
            login, password, ip, port = line.split(":")
            formatted.append(f"http://{login}:{password}@{ip}:{port}")
        else:
            formatted.append(line)
    return formatted

def worker(index, count, delay_range, keys_cycle, proxy, system_prompt, lang, max_tokens):
    for i in range(count):
        api_key = next(keys_cycle)
        question = generate_question(api_key, system_prompt, max_tokens, proxy, lang)
        if question:
            console.print(f"🟡 Поток {index} [{i+1}/{count}] Вопрос: {question} Прокси: {proxy or 'none'}")
            send_prompt(question, api_key, system_prompt, max_tokens, proxy)
        if i < count - 1:
            sleep_time = random.uniform(delay_range[0], delay_range[1])
            console.print(f"⏳ Поток {index} пауза {sleep_time:.2f} сек\n{'-'*40}", style="dim")
            time.sleep(max(0.1, sleep_time))

def get_user_inputs():
    try:
        count = int(input("🔢 Введите количество запросов (default 5): ") or 5)
        delay_input = input("⏱ Введите задержку между запросами в секундах (например '1 3', default 3): ") or "3"
        delay_parts = delay_input.strip().split()

        if len(delay_parts) == 2:
            delay_min, delay_max = float(delay_parts[0]), float(delay_parts[1])
            if delay_min > delay_max:
                delay_min, delay_max = delay_max, delay_min
        else:
            delay_min = delay_max = float(delay_parts[0])
        max_tokens = input("Введи максимальное количество используемых токенов (default 128): ") or 128
        threads = int(input("🧵 Введите количество потоков (default 1): ") or 1)
        lang_choice = input("🌍 Выберите язык (r=русский, e=english, b=оба, default r): ").strip().lower() or 'r'

        proxy_choices = []
        for i in range(threads):
            use_proxy = input(f"🌐 Использовать прокси для потока {i+1}? (y/n, default n): ").strip().lower()
            proxy_choices.append(use_proxy == 'y')

        return count, (delay_min, delay_max), threads, lang_choice, proxy_choices, max_tokens
    except Exception:
        console.print("❌ Некорректный ввод. Попробуйте снова.", style="red")
        return get_user_inputs()

def main():
    print_banner()
    count, delay_range, threads, lang_choice, proxy_choices, max_tokens = get_user_inputs()

    console.print(f"\nПараметры запуска:\n- Запросов на поток: {count}\n- Задержка: от {delay_range[0]} до {delay_range[1]} сек\nМаксимальная количество используемых токенов до {max_tokens}\n- Потоков: {threads}\n- Язык: {lang_choice}\n", style="cyan")

    input("Нажмите Enter для запуска...")

    api_keys = load_lines("API_keys.txt")
    if not api_keys:
        return console.print("❌ Нет API ключей.", style="red")

    system_prompt = "\n".join(load_lines("promt.txt")) or "You are a helpful assistant."

    raw_proxies = load_lines("proxy.txt")
    formatted_proxies = format_proxies(raw_proxies)
    proxy_pool = cycle(formatted_proxies) if formatted_proxies else None

    keys_cycle = cycle(api_keys)

    threads_list = []
    for i in range(threads):
        proxy = next(proxy_pool) if proxy_choices[i] and proxy_pool else None
        t = threading.Thread(target=worker, args=(i+1, count, delay_range, keys_cycle, proxy, system_prompt, lang_choice, max_tokens))
        t.start()
        threads_list.append(t)

    for t in threads_list:
        t.join()

if __name__ == "__main__":
    main()
