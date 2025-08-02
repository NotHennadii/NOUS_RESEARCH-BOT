#!/bin/bash

# Устанавливаем заголовок терминала
echo -e "\033]0;NOUS RESEARCH\007"

# Цветной вывод
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 NOUS RESEARCH LAUNCHER for macOS${NC}"
echo "=================================="

# Проверяем наличие Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 не найден!${NC}"
    echo -e "${YELLOW}Установите Python 3 через Homebrew: brew install python${NC}"
    echo -e "${YELLOW}Или скачайте с https://www.python.org/downloads/${NC}"
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

echo -e "${GREEN}✅ Python 3 найден: $(python3 --version)${NC}"

# Создаем виртуальное окружение если его нет
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Создаю виртуальное окружение...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Ошибка создания виртуального окружения${NC}"
        read -p "Нажмите Enter для выхода..."
        exit 1
    fi
fi

# Активируем виртуальное окружение
echo -e "${YELLOW}🔧 Активирую виртуальное окружение...${NC}"
source venv/bin/activate

# Устанавливаем зависимости
echo -e "${YELLOW}📚 Устанавливаю зависимости...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Ошибка установки зависимостей${NC}"
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

# Проверяем наличие нужных файлов
if [ ! -f "NOUS.py" ]; then
    echo -e "${RED}❌ Файл NOUS.py не найден!${NC}"
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

if [ ! -f "API_keys.txt" ]; then
    echo -e "${YELLOW}⚠️  Файл API_keys.txt не найден. Создаю пример...${NC}"
    echo "# Добавьте ваши API ключи сюда, по одному на строку" > API_keys.txt
    echo "# your_api_key_1" >> API_keys.txt
    echo "# your_api_key_2" >> API_keys.txt
    echo -e "${BLUE}💡 Отредактируйте файл API_keys.txt и добавьте ваши ключи${NC}"
fi

if [ ! -f "promt.txt" ]; then
    echo -e "${YELLOW}⚠️  Файл promt.txt не найден. Создаю пример...${NC}"
    echo "Ты умный AI ассистент. Отвечай кратко и по делу." > promt.txt
fi

if [ ! -f "proxy.txt" ]; then
    echo -e "${YELLOW}⚠️  Файл proxy.txt не найден. Создаю пустой файл...${NC}"
    touch proxy.txt
fi

# Запускаем скрипт
echo -e "${GREEN}🎯 Запускаю NOUS.py...${NC}"
echo "=================================="
python3 NOUS.py

# Пауза после завершения
echo ""
echo -e "${BLUE}Программа завершена.${NC}"
read -p "Нажмите Enter для выхода..."