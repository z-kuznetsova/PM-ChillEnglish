import string  # Импортируем модуль для работы со строками
import nltk  # Импортируем библиотеку для обработки естественного языка
from sklearn.feature_extraction.text import TfidfVectorizer  # Импортируем векторизатор TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # Импортируем функцию для вычисления косинусного сходства
from nltk.corpus.reader.plaintext import PlaintextCorpusReader  # Импортируем класс для чтения текстовых корпусов
import natasha as nt  # Импортируем библиотеку Natasha для обработки русского языка
from telegram import Update  # Импортируем класс Update из библиотеки telegram
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes  # Импортируем необходимые классы для работы с Telegram API

# Загрузите необходимые ресурсы NLTK
nltk.download('punkt')  # Загружаем ресурс для токенизации текста

# Вспомогательные функции
def Normalize(text):
    # Инициализируем необходимые компоненты Natasha
    segmenter = nt.Segmenter()  # Сегментатор текста
    morph_vocab = nt.MorphVocab()  # Морфологический словарь
    emb = nt.NewsEmbedding()  # Эмбеддинги на основе новостных текстов
    morph_tagger = nt.NewsMorphTagger(emb)  # Морфологический теггер
    ner_tagger = nt.NewsNERTagger(emb)  # Теггер для именованных сущностей

    # Удаляем знаки препинания и создаем токены
    word_token = text.translate(str.maketrans("", "", string.punctuation)).replace("—", "")
    doc = nt.Doc(word_token)  # Создаем документ с токенами
    doc.segment(segmenter)  # Сегментируем текст
    doc.tag_morph(morph_tagger)  # Тегируем морфологию
    doc.tag_ner(ner_tagger)  # Тегируем именованные сущности

    # Лемматизация токенов
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    resDict = {_.text: _.lemma for _ in doc.tokens}  # Создаем словарь токенов и их лемм
    return [resDict[i] for i in resDict]  # Возвращаем список лемм

def Response(user_response):
    # Приводим ответ пользователя к нижнему регистру
    user_response = user_response.lower()
    robo_response = ''  # Инициализируем переменную для ответа бота
    sent_tokens.append(user_response)  # Добавляем ответ пользователя в список токенов
    TfidfVec = TfidfVectorizer(tokenizer=Normalize)  # Инициализируем векторизатор с нормализацией
    tfidf = TfidfVec.fit_transform(sent_tokens)  # Преобразуем токены в TF-IDF матрицу
    vals = cosine_similarity(tfidf[-1], tfidf)  # Вычисляем косинусное сходство
    idx = vals.argsort()[0][-2]  # Находим индекс второго по значимости ответа
    flat = vals.flatten()  # Преобразуем матрицу в одномерный массив
    flat.sort()  # Сортируем массив
    req_tfidf = flat[-2]  # Получаем значение второго по значимости TF-IDF
    sent_tokens.remove(user_response)  # Удаляем ответ пользователя из списка токенов

    # Формируем ответ бота
    if req_tfidf == 0:
        robo_response = "Извините, я не нашел ответа..."
    else:
        robo_response = sent_tokens[idx]  # Возвращаем наиболее подходящий ответ
    return robo_response

# Инициализация корпуса
newcorpus = PlaintextCorpusReader('newcorpus/', r'.*\.txt')  # Читаем текстовый корпус из указанной директории
data = newcorpus.raw(newcorpus.fileids())  # Получаем сырые данные из корпуса
sent_tokens = nltk.sent_tokenize(data)  # Токенизируем текст на предложения

# Списки приветствий и прощаний
welcome_input = ["привет", "ку", "прив", "добрый день", "доброго времени суток", "здравствуйте", "приветствую"]
goodbye_input = ["пока", "стоп", "выход", "конец", "до свидания"]

# Функция для обработки команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот, готовый ответить на ваши вопросы.")  # Ответ на команду /start

# Функция для обработки текстовых сообщений
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.lower()  # Получаем текст сообщения от пользователя

    # Проверяем, содержит ли сообщение приветствие или прощание
    if any(word in user_message for word in welcome_input):
        await update.message.reply_text("Привет!")  # Ответ на приветствие
    elif any(word in user_message for word in goodbye_input):
        await update.message.reply_text("Буду ждать вас!")  # Ответ на прощание
    else:
        response = Response(user_message)  # Генерируем ответ с помощью функции Response
        await update.message.reply_text(response)  # Отправляем ответ пользователю

# Основная функция для запуска бота
def main():
    # Замените 'YOUR_TOKEN' на ваш токен бота
    application = ApplicationBuilder().token("7706748289:AAFOov2786JN6EryOQZTUeow6Fx1rXfw2Qc").build()  # Создаем приложение Telegram

    # Добавляем обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))  # Обработчик команды /start
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))  # Обработчик текстовых сообщений

    application.run_polling()  # Запускаем бота в режиме опроса

if __name__ == '__main__':
    main()  # Запускаем основную функцию
