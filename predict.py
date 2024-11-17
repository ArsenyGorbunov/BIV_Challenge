"""
Классификатор платежей
---------------------
Скрипт для обработки платежных данных и предсказания их классификации с использованием
ансамбля моделей CatBoost. Включает предобработку текста, лемматизацию и функционал
для предсказаний.

Проект команды: Команда10
Дата: Ноябрь 2024
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from loguru import logger
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from nltk.corpus import stopwords
import nltk
import pymorphy2
import re
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Конфигурация путей к файлам и параметров модели."""
    
    data_path: Path = Path('./data/payments_main.tsv')
    models_dir: Path = Path('./models')
    output_dir: Path = Path('./predictions')
    log_file: Path = Path('./logs/processing.log')
    
    # Параметры модели
    n_models: int = 5
    model_weights: np.ndarray = field(default_factory=lambda: np.array([
        0.20274677, 0.21557854, 0.17898084, 0.21226201, 0.19043183
    ]))


class TextPreprocessor:
    """Обработка текстовых данных."""
    
    def __init__(self):
        """Инициализация инструментов обработки текста."""
        self._initialize_nltk()
        self._initialize_natasha()
        self.months = self._get_month_list()
        
    def _initialize_nltk(self) -> None:
        """Инициализация ресурсов NLTK."""
        try:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words("russian"))
        except Exception as e:
            logger.error(f"Ошибка инициализации NLTK: {str(e)}")
            raise
            
    def _initialize_natasha(self) -> None:
        """Инициализация инструментов Natasha."""
        self.morph_vocab = MorphVocab()
        self.segmenter = Segmenter()
        self.morph_tagger = NewsMorphTagger(NewsEmbedding())
        self.morph_analyzer = pymorphy2.MorphAnalyzer()
        
    @staticmethod
    def _get_month_list() -> List[str]:
        """Возвращает список названий месяцев на русском языке."""
        return [
            "январь", "февраль", "март", "апрель", "май", "июнь", 
            "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
            "января", "февраля", "марта", "апреля", "мая", "июня", 
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]
        
    def preprocess_text(self, text: str) -> str:
        """
        Очистка и нормализация текста.
        
        Аргументы:
            text: Входной текст
            
        Возвращает:
            Обработанный текст
        """
        if not isinstance(text, str):
            return text
            
        try:
            text = text.lower()
            text = self._remove_urls_and_emails(text)
            text = self._remove_numbers_and_punctuation(text)
            text = self._remove_months(text)
            text = self._normalize_abbreviations(text)
            text = self._filter_russian_words(text)
            text = self._remove_stopwords(text)
            return text.strip()
        except Exception as e:
            logger.error(f"Ошибка предобработки текста: {str(e)}")
            return text
            
    def lemmatize_text(self, text: str) -> str:
        """
        Лемматизация текста с помощью Natasha.
        
        Аргументы:
            text: Входной текст
            
        Возвращает:
            Лемматизированный текст
        """
        try:
            text = ' '.join(text.split('-'))
            text = ' '.join(word for word in text.split() if len(word) > 1)
            
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)
                
            tokens = [
                t for t in doc.tokens 
                if t.pos not in ['PUNCT', 'PRON', 'DET', 'CCONJ', 'INTJ', 'PRCL', 'ADP'] 
                and t.lemma not in self.stopwords
            ]
            
            return ' '.join(t.lemma for t in tokens)
        except Exception as e:
            logger.error(f"Ошибка лемматизации: {str(e)}")
            return text
            
    @staticmethod
    def _remove_urls_and_emails(text: str) -> str:
        """Удаление URL и email адресов из текста."""
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
        return text
        
    @staticmethod
    def _remove_numbers_and_punctuation(text: str) -> str:
        """Удаление чисел и знаков препинания."""
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text)
        
    def _remove_months(self, text: str) -> str:
        """Удаление названий месяцев из текста."""
        for month in self.months:
            text = re.sub(rf'\b{month}\b', '', text)
        return text
        
    @staticmethod
    def _normalize_abbreviations(text: str) -> str:
        """Нормализация распространенных сокращений."""
        text = re.sub(r'\bгос\.?\b', 'государственный', text)
        text = re.sub(r'\bдог\.?\b', 'договор', text)
        return text
        
    @staticmethod
    def _filter_russian_words(text: str) -> str:
        """Оставляет только русские слова."""
        words = text.split()
        russian_words = [word for word in words if re.match(r'^[а-яё]+$', word)]
        return ' '.join(russian_words)
        
    def _remove_stopwords(self, text: str) -> str:
        """Удаление стоп-слов русского языка."""
        words = text.split()
        return ' '.join(word for word in words if word not in self.stopwords)


class PaymentClassifier:
    """Работа с ансамблем моделей и получение предсказаний."""
    
    def __init__(self, config: ModelConfig):
        """
        Инициализация классификатора.
        
        Аргументы:
            config: Экземпляр ModelConfig с путями и параметрами
        """
        self.config = config
        self.models = []
        self.label_encoder = None
        self._load_models()
        
    def _load_models(self) -> None:
        """Загрузка моделей ансамбля и энкодера меток."""
        try:
            for i in range(self.config.n_models):
                model = CatBoostClassifier()
                model.load_model(self.config.models_dir / f'ensemble_model_{i}.cbm')
                self.models.append(model)
                
            with open(self.config.models_dir / 'label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {str(e)}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получение предсказаний ансамбля.
        
        Аргументы:
            X: DataFrame с признаками
            
        Возвращает:
            Массив предсказанных меток
        """
        try:
            n_classes = len(self.models[0].classes_)
            ensemble_probs = np.zeros((len(X), n_classes))
            
            for model, weight in zip(self.models, self.config.model_weights):
                probs = model.predict_proba(X)
                smoothed_probs = (probs + 0.001) / (1 + 0.001 * n_classes)
                ensemble_probs += weight * smoothed_probs
                
            predictions = np.argmax(ensemble_probs / np.sum(self.config.model_weights), axis=1)
            return self.label_encoder.inverse_transform(predictions)
        except Exception as e:
            logger.error(f"Ошибка предсказания: {str(e)}")
            raise


class PaymentProcessor:
    """Основной класс для обработки платежей и получения предсказаний."""
    
    def __init__(self, config: ModelConfig):
        """
        Инициализация процессора.
        
        Аргументы:
            config: Экземпляр ModelConfig
        """
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.classifier = PaymentClassifier(config)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Настройка логирования."""
        log_format = "{time} {level} {message}"
        logger.add(self.config.log_file, format=log_format, level="INFO", rotation="10 MB")
        
    def load_data(self) -> pd.DataFrame:
        """
        Загрузка и валидация данных о платежах.
        
        Возвращает:
            DataFrame с данными о платежах
        """
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.config.data_path}")
            
        try:
            df = pd.read_csv(self.config.data_path, sep="\t", header=None)
            df.columns = ['id', 'date', 'sum', 'describe']
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {str(e)}")
            raise
            
    def process_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка текстовых описаний в DataFrame.
        
        Аргументы:
            df: Входной DataFrame
            
        Возвращает:
            DataFrame с обработанным текстом
        """
        logger.info("Обработка текстовых описаний...")
        tqdm.pandas()
        
        try:
            df['lemmed_opisanie'] = df['describe'].fillna('<Nan>').progress_apply(
                lambda x: self.preprocessor.lemmatize_text(x) if len(x) > 0 else '<Nan>'
            )
            df['lemmed_opisanie'] = df['lemmed_opisanie'].apply(self.preprocessor.preprocess_text)
            return df[['id', 'lemmed_opisanie']]
        except Exception as e:
            logger.error(f"Ошибка обработки текста: {str(e)}")
            raise
            
    def save_predictions(self, ids: np.ndarray, predictions: np.ndarray) -> None:
        """
        Сохранение предсказаний в файл.
        
        Аргументы:
            ids: Массив идентификаторов платежей
            predictions: Массив предсказанных меток
        """
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            predict_df = pd.DataFrame({'id': ids, 'predictions': predictions})
            output_path = self.config.output_dir / 'predict.tsv'
            predict_df.to_csv(output_path, sep='\t', index=False, header=False)
            logger.success(f"Предсказания сохранены в {output_path}")
            # В методе save_predictions после сохранения файла
            logger.info(f"Содержимое директории predictions: {list(self.config.output_dir.iterdir())}")
        except Exception as e:
            logger.error(f"Ошибка сохранения предсказаний: {str(e)}")
            raise


def main():
    """Основная функция выполнения."""
    try:
        # Инициализация конфигурации и процессора
        config = ModelConfig()
        processor = PaymentProcessor(config)
        
        # Загрузка и обработка данных
        logger.info("Начало процесса классификации платежей...")
        df = processor.load_data()
        processed_df = processor.process_text(df)
        
        # Получение предсказаний
        predictions = processor.classifier.predict(processed_df.drop(columns='id'))
        
        # Сохранение результатов
        processor.save_predictions(processed_df['id'].values, predictions)
        logger.success("Классификация платежей успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {str(e)}")
        raise


if __name__ == "__main__":
    main()