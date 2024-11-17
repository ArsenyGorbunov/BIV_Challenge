# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from loguru import logger
import pymorphy2
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from nltk.corpus import stopwords
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize

tqdm.pandas(desc="bar")
# Настройка логирования
logger.add("pipeline.log", rotation="500 MB", level="INFO")


class PaymentProcessor:
    def __init__(self, data_path='./data/'):
        """
        Инициализация класса для обработки платежных данных
        """
        self.PATH = data_path
        self.morph_vocab = MorphVocab()

        # Словарь для конвертации тегов OpenCorpora в Universal Tags
        self.OC_UD_POS = {
            'ADJF': 'ADJ', 'ADJS': 'ADJ',  # Прилагательные
            'ADVB': 'ADV', 'COMP': 'ADV',  # Наречия
            'VERB': 'VERB', 'GRND': 'VERB',  # Глаголы
            'INFN': 'VERB', 'PRTF': 'VERB',
            'PRTS': 'VERB',
            'NOUN': 'NOUN',  # Существительные
            'NPRO': 'PRON',  # Местоимения
            'NUMR': 'NUM', 'NUMB': 'NUM',  # Числительные
            'Apro': 'DET',  # Местоименные прилагательные
            'CONJ': 'CCONJ',  # Союзы
            'INTJ': 'INTJ',  # Междометия
            'PART': 'PRCL',  # Частицы
            'PNCT': 'PUNCT',  # Пунктуация
            'PRCL': 'PART',  # Частицы
            'PREP': 'ADP',  # Предлоги
        }

        # Теги частей речи для исключения
        self.STOP_TAGS = ['PRON', 'DET', 'CCONJ', 'INTJ', 'PRCL', 'ADP']

        # Загрузка стоп-слов
        import nltk
        nltk.download('stopwords', quiet=True)
        self.russian_stopwords = stopwords.words("russian")

        # Список месяцев для удаления
        self.months = [
            "январь", "февраль", "март", "апрель", "май", "июнь",
            "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря"
        ]

    def load_data(self):
        """
        Загрузка и первичная обработка данных
        """
        logger.info("Начало загрузки данных")

        # Загрузка основного датасета
        self.payments_main = pd.read_csv(self.PATH + 'payments_main.tsv',
                                         sep="\t",
                                         header=None)
        self.payments_main.columns = ['id', 'date', 'sum', 'describe']

        # Загрузка тренировочного датасета
        self.payments_training = pd.read_csv(self.PATH + 'payments_training.tsv',
                                             sep="\t",
                                             header=None)
        self.payments_training.columns = ['id', 'date', 'sum', 'describe', 'label']

        logger.info(f"Загружено записей: основной датасет - {self.payments_main.shape}, "
                    f"тренировочный датасет - {self.payments_training.shape}")

    def lemmatize_text(self, text,
                       segmenter=Segmenter(),
                       morph_tagger=NewsMorphTagger(NewsEmbedding()),
                       pyMorphyMorphAnalyser=pymorphy2.MorphAnalyzer()):
        """
        Лемматизация текста с использованием Natasha и pymorphy2
        """
        if not isinstance(text, str) or len(text) == 0:
            return '<Nan>'

        # Предобработка текста
        text = ' '.join(text.split('-'))
        text = ' '.join(i for i in text.split() if len(i) > 1)

        # Создание документа и его обработка
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        # Обработка токенов
        for token in doc.tokens:
            pm_phorms = [x for x in pyMorphyMorphAnalyser.parse(token.text)]
            tags = self._process_morphological_tags(pm_phorms)
            token.lemmatize(self.morph_vocab)

        # Фильтрация токенов
        tokens = [t for t in doc.tokens
                  if t.pos != 'PUNCT' and
                  t.lemma not in self.russian_stopwords and
                  t.pos not in self.STOP_TAGS]

        return ' '.join([t.lemma for t in tokens])

    def _process_morphological_tags(self, pm_phorms):
        """
        Обработка морфологических тегов
        """
        return [[str(x.tag).split(',')[0].split(' ')[0],
                 self.OC_UD_POS.get(str(x.tag).split(',')[0].split(' ')[0]),
                 x.normal_form]
                for x in pm_phorms]

    def preprocess_text(self, text):
        """
        Предобработка текстовых данных
        """
        if not isinstance(text, str):
            return text

        # Последовательная обработка текста
        text = text.lower()
        text = self._remove_special_elements(text)
        text = self._process_abbreviations(text)
        text = self._filter_words(text)

        return text

    def _remove_special_elements(self, text):
        """
        Удаление специальных элементов из текста
        """
        patterns = [
            (r'https?://\S+|www\.\S+', ''),  # URLs
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ''),  # Emails
            (r'\d+', ''),  # Digits
            (r'[^\w\s]', ' '),  # Special characters
            (r'\s+', ' '),  # Extra spaces
            (r'^(fw|re|re:)', '', re.IGNORECASE),  # Email prefixes
            (r'\b\w\b', '')  # Single characters
        ]

        for pattern, repl, *flags in patterns:
            text = re.sub(pattern, repl, text, *flags)

        return text.strip()

    def _process_abbreviations(self, text):
        """
        Обработка сокращений
        """
        replacements = {
            r'\bгос\.?\b': 'государственный',
            r'\bдог\.?\b': 'договор'
        }

        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)

        return text

    def _filter_words(self, text):
        """
        Фильтрация слов
        """
        # Удаление названий месяцев
        for month in self.months:
            text = re.sub(rf'\b{month}\b', '', text)

        # Фильтрация по кириллице
        words = text.split()
        words = [word for word in words if re.match(r'^[а-яё]+$', word)]
        words = [word for word in words if word not in self.russian_stopwords]

        return ' '.join(words)

    def process_datasets(self):
        """
        Обработка обоих датасетов
        """
        logger.info("Начало обработки датасетов")

        # Обработка тренировочного датасета
        self.payments_training['lemmed_opisanie'] = (
            self.payments_training['describe']
                .fillna('<Nan>')
                .progress_apply(self.lemmatize_text)
                .apply(self.preprocess_text)
        )

        # Обработка основного датасета
        self.payments_main['lemmed_opisanie'] = (
            self.payments_main['describe']
                .fillna('<Nan>')
                .progress_apply(self.lemmatize_text)
                .apply(self.preprocess_text)
        )

        # Обработка числовых значений
        for df in [self.payments_training, self.payments_main]:
            df['sum'] = (df['sum']
                         .str.replace(',', '')
                         .str.replace('-', '')
                         .apply(lambda x: float(x) if '.' in x else int(x))
                         .astype(int))

        logger.info("Обработка датасетов завершена")


class ModelTrainer:
    def __init__(self):
        """
        Инициализация класса для обучения моделей
        """
        self.catboost_params = {
            'iterations': 5_000,
            'learning_rate': 0.01,
            'eval_metric': 'Accuracy',
            'early_stopping_rounds': 1000,
            'use_best_model': True,
            'verbose': False,
            'random_seed': 42
        }

        self.text_cols = ['lemmed_opisanie']
        self.embedding_features = []
        self.categorical_cols = []

    def train_initial_model(self, X_train, y_train, X_test):
        """
        Обучение начальной модели и получение псевдо-меток
        """
        logger.info("Начало обучения начальной модели")

        # Создание пулов данных
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=self.categorical_cols,
            text_features=self.text_cols
        )

        # Обучение модели
        model = CatBoostClassifier(**self.catboost_params)
        model.fit(train_pool, eval_set=train_pool, plot=False)

        # Сохранение модели
        os.makedirs('../models', exist_ok=True)
        model.save_model('./models/catboost_model.cbm')

        # Получение предсказаний
        y_pred_proba = model.predict_proba(X_test)

        logger.info("Начальная модель обучена и сохранена")
        return model, y_pred_proba

    def create_pseudo_labels(self, y_pred_proba, y_pred, X_test, threshold=0.8):
        """
        Создание псевдо-меток на основе предсказаний модели
        """
        logger.info(f"Создание псевдо-меток с порогом {threshold}")

        # Фильтрация по уверенности предсказаний
        max_values = np.max(y_pred_proba, axis=1)
        indices = np.where(max_values > threshold)[0]

        # Выбор отфильтрованных данных
        filtered_predictions = pd.Series(
            y_pred[indices].flatten(),
            name="Predicted_Class"
        )
        filtered_features = X_test.iloc[indices].reset_index(drop=True)

        logger.info(f"Создано {len(indices)} псевдо-меток")
        return filtered_features, filtered_predictions

    def train_ensemble(self, X_combined, y_combined, n_folds=5, n_classes=9):
        """
        Обучение ансамбля моделей с использованием кросс-валидации
        """
        logger.info(f"Начало обучения ансамбля из {n_folds} моделей")

        # Инициализация
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_combined)

        # Подготовка кросс-валидации
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        models = []
        fold_predictions = []
        fold_indices = []

        # Обучение моделей на фолдах
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_encoded)):
            logger.info(f"Обучение модели для фолда {fold_idx + 1}")

            # Разделение данных
            X_tr, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
            y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

            # Создание пулов данных
            train_pool = Pool(
                X_tr,
                y_tr,
                text_features=self.text_cols,
                cat_features=self.categorical_cols
            )

            eval_pool = Pool(
                X_val,
                y_val,
                text_features=self.text_cols,
                cat_features=self.categorical_cols
            )

            # Обучение модели
            model = CatBoostClassifier(**self.catboost_params)
            model.fit(train_pool, eval_set=eval_pool, verbose=False)

            # Сохранение результатов
            models.append(model)
            fold_predictions.append(model.predict_proba(X_val))
            fold_indices.append(val_idx)

            # Оценка качества
            accuracy = accuracy_score(y_val, model.predict(X_val))
            logger.info(f"Fold {fold_idx + 1} Accuracy: {accuracy:.4f}")

        # Оптимизация весов ансамбля
        optimal_weights = self._optimize_ensemble_weights(
            models, fold_predictions, y_encoded, fold_indices, n_classes
        )
        # Оптимизация весов ансамбля
        optimal_weights = self._optimize_ensemble_weights(
            models, fold_predictions, y_encoded, fold_indices, n_classes
        )

        # Сохранение моделей ансамбля
        logger.info("Сохранение моделей ансамбля")
        os.makedirs('../models', exist_ok=True)
        for i, model in enumerate(models):
            model.save_model(f'./models/ensemble_model_{i}.cbm')

        logger.info("Обучение ансамбля завершено")
        return {
            'models': models,
            'weights': optimal_weights,
            'label_encoder': label_encoder,
            'fold_indices': fold_indices
        }

    def _optimize_ensemble_weights(self, models, predictions, y_true, indices, n_classes):
        """
        Оптимизация весов для ансамбля моделей
        """
        logger.info("Начало оптимизации весов ансамбля")

        def objective_function(weights, all_preds, y_true, indices):
            """
            Целевая функция для оптимизации весов
            """
            # Нормализация весов
            weights = weights / np.sum(weights)

            # Инициализация массивов для взвешенных предсказаний
            all_predictions = np.zeros((len(y_true), n_classes))
            counts = np.zeros(len(y_true))

            # Расчет взвешенных предсказаний
            for fold_idx, (preds, idx) in enumerate(zip(all_preds, indices)):
                all_predictions[idx] += weights[fold_idx] * preds
                counts[idx] += weights[fold_idx]

            # Нормализация и получение финальных предсказаний
            all_predictions = all_predictions / counts[:, np.newaxis]
            ensemble_preds = np.argmax(all_predictions, axis=1)

            return -f1_score(y_true, ensemble_preds, average='weighted')

        # Начальные веса и ограничения
        n_folds = len(models)
        initial_weights = np.ones(n_folds) / n_folds
        bounds = [(0, 1)] * n_folds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Оптимизация
        result = minimize(
            objective_function,
            initial_weights,
            args=(predictions, y_true, indices),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        optimal_weights = result.x / np.sum(result.x)
        ensemble_score = -result.fun

        logger.info(f"Оптимизация завершена. F1-score ансамбля: {ensemble_score:.4f}")
        logger.info(f"Оптимальные веса: {[f'{w:.4f}' for w in optimal_weights]}")

        return optimal_weights

    def get_ensemble_predictions(self, models, weights, X, label_encoder=None):
        """
        Получение предсказаний от ансамбля моделей
        """
        logger.info("Получение предсказаний ансамбля")

        n_classes = len(models[0].classes_)
        ensemble_probs = np.zeros((len(X), n_classes))

        # Расчет взвешенных вероятностей
        for model, weight in zip(models, weights):
            # Получение и сглаживание вероятностей
            model_probs = model.predict_proba(X)
            smoothed_probs = (model_probs + 0.001) / (1 + 0.001 * n_classes)
            ensemble_probs += weight * smoothed_probs

        # Получение финальных предсказаний
        predictions = np.argmax(ensemble_probs / np.sum(weights), axis=1)

        # Преобразование числовых меток обратно в строковые, если необходимо
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)

        logger.info("Предсказания ансамбля получены")
        return predictions


def main():
    """
    Основная функция для запуска всего пайплайна
    """
    logger.info("Запуск пайплайна обработки платежей и обучения моделей")

    # Инициализация процессора данных
    processor = PaymentProcessor()

    # Загрузка и обработка данных
    processor.load_data()
    processor.process_datasets()

    # Подготовка данных для обучения
    X_train_full = pd.DataFrame(processor.payments_training['lemmed_opisanie'])
    y_train_full = processor.payments_training['label']
    X_test = pd.DataFrame(processor.payments_main['lemmed_opisanie'])

    logger.info(f"Подготовлены данные: X_train shape: {X_train_full.shape}, "
                f"X_test shape: {X_test.shape}")

    # Инициализация и обучение моделей
    trainer = ModelTrainer()

    # Обучение начальной модели
    initial_model, y_pred_proba = trainer.train_initial_model(
        X_train_full,
        y_train_full,
        X_test
    )

    # Создание псевдо-меток
    X_test_filtered, y_pred_filtered = trainer.create_pseudo_labels(
        y_pred_proba,
        initial_model.predict(X_test),
        X_test
    )

    # Объединение данных для обучения ансамбля
    X_combined = pd.concat([X_train_full, X_test_filtered], axis=0)
    y_combined = pd.concat([y_train_full, y_pred_filtered], axis=0)

    logger.info(f"Данные для ансамбля: shape {X_combined.shape}")

    # Сохранение объединенных данных
    X_combined.to_parquet(processor.PATH + 'X_combined.parquet', index=False)
    y_combined.to_frame(name='y_combined').to_parquet(
        processor.PATH + 'y_combined.parquet',
        index=False
    )

    # Обучение ансамбля моделей
    ensemble_results = trainer.train_ensemble(
        X_combined,
        y_combined,
        n_folds=5,
        n_classes=len(np.unique(y_combined))  # Автоматический подсчет уникальных классов
    )

    logger.info("Пайплайн успешно завершен")


if __name__ == "__main__":
    main()
