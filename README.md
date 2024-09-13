PDResearch 
Проект повящен исследованию двигательной активности у пациентов с болезнью Паркинсона.

Актуальные задачи:
1. ML & DL. Разработка алгоритма многоклассовой классификации и алгоритма оценки тяжести заболевания по оценкам MDS-UPDRS и/или стадиям заболевания, на основе рассчитанных признаков, AutoML, Optuna, классическое машинное обучение и MLP, генерация синтетических данных (SMOTE,VAE), снижение размерности (PCA, AE), кластеризация
3. ML & DL. Разработка алгоритма оценки тяжести заболевания по оценкам MDS-UPDRS и/или стадиям заболевания на основе сырых сигналов двигатльной активности рук, нейронная сеть (RNN, CNN), MLFlow, W&B
3. Computer vision. Разработка алгоритма оценки параметров двигательной активности рук (постукивание пальцами, открытие/закрытие ладони) по 2D видеоизображению и алгоритма классификации/оценки тяжести заболевания на основе нейронной сети (CNN), MLFlow, W&B
4. Signal processing & Computer vision. Разработка алгоритма оценки параметров двигательной активности рук (постукивание пальцами, открытие/закрытие ладони) по 2D видеоизображению и алгоритма классификации/оценки тяжести заболевания на основе построения сигналов с MediaPipe (через расстояния и углы между ключевыми точками)
5. Computer vision. Разработка алгоритма оценки параметров двигательной активности рук (постукивание пальцами, открытие/закрытие ладони) по 2D видеоизображению и алгоритма классификации/оценки тяжести заболевания с использованием цветовых маркеров/насечек на ключевых точках ладони 
6. Signal processing. Разработка алгоритма оценки параметров двигательной активности рук (пронация/супинация кисти) на основе анализа сигналов с акселерометра смартфона, разработка алгоритма классификации/оценки тяжести заболевания 
7. Signal processing. Разработка алгоритма оценки параметров тремора рук с датчика LeapMotion (3D камера) и 2D камеры 
8. Signal processing. Разработка алгоритма оценки параметров голоса, разработка алгоритма классификации/оценки тяжести заболевания 
9. Signal processing. Разработка алгоритма автоматизированной разметки сигналов двигательной активности
9. Dev. Разработка веб-сервиса для задачи анализа мимики по 2D и задачи 3,4 
10. Dev. Разработка мобильного приложения под Android для задачи 5 
11. Dev. Разработка мобильного приложения под Android для задачи 3,4 
12. Dev. Разработка десктопного приложения под Windows для задачи анализа мимики и двигательной активности рук



Методика регистрации данных:
1. Дигательная активность рук с LeapMotion (3D камера) - 2 раза
   - Постукивание пальцами "FT" - запись по очереди правая, левая рука в течении 20 секунд
   - Открытие/зактытие ладони "OC" - запись по очереди правая, левая рука в течении 20 секунд
   - Пронация/супинация кисти "PS" - запись по очереди правая, левая рука в течении 20 секунд
   - Тремор (TR) - запись двух рук одновременно в течении 20 секунд
2. Мимическая актвность (2D камера) - 2 раза
    - улыбка с усилием 10 раз (AU12) - p5
    - нахмурить брови 10 раз (AU04) - p11
3. Лицевая выразительность (2D камера) - 2 раза
    - нейтральное выражение лица 10 секунд
    - повторить выражение лица на изображении (6 базовых эмоций) 2 секунды

Расширение методики:
4. Дигательная активность рук (2D камера) - 1 раз
   - Постукивание пальцами "FT" - запись по очереди правая, левая рука в течении 20 секунд
   - Открытие/зактытие ладони "OC" - запись по очереди правая, левая рука в течении 20 секунд
   - Тремор (TR) - запись двух рук одновременно в течении 20 секунд
5. Дигательная активность рук (SmarPhone, акселерометр) - 1 раз
   - Пронация/супинация кисти "PS" с телефоном в руке - запись по очереди правая, левая рука в течении 20 секунд
6. Запись речи
    - Чтение текста
    - Повторение аааааа

Структура хранения данных:
    
    Папка - id участника "Patient1"
            - r0 - номер эксперимента
                - face - все видео с мимикой и их обработанными данными (пункт 2,3)
                - m1 - папка с записями lmt рук (пункт 1)
                - hand - обработанные данные m папок
                - hand2D - видео и lmt (пункт 4)
                - handAcc - файлы с телефона и lmt (пункт 5)
                - woice - файлы голоса wav с телефона
    Для одного участника может быть несколько папок с r, т.е. он может быть записан нескольо раз по одной и той же методике

Очередность выполнения:
1. Регистрация данных по методике
2. Проверка очередности/порядка следования видео, количества данных, правильная организация хранения
3. Внесение сведений об испытуемом в таблицу (номер, ФИО, возраст, пол, стадия и др.)
5. Обработка данных - запуск пайплайна 'processing'
7. Проверка данных  - запуск пайплайна 'check_data'
8. Внесение сведений о данных в таблицу (качество, комментарии)
6. Ручная разметка данных с использованием приложения для разметки данных (Programms/application)
7. Создание папок с данными ручной разметки 'mannual_point' - для рук, и 'mannual_point_face' - для лица

Необходимые 'exe' для обработки:

      - двигательная активность рук, тремор *https://github.com/AnastasiaMoshkova/LeapMotionPlayback* (чтение '.lmt')
      - мимика OpenFace
      - приложение для разметки данных - скоро на githab (пока что бинарники в архиве на гугл диске)

Проект:

    Проект выполнени с исполользованием конфигураций к кажому эксперименту - конфигурации находятся в папке 'configs'
    
    Поддерживаются следюущие пайплайны (в порядке очередности выполнения):

        1. Предобработка данных (преобразование сырых данных в сигналы для обработки) preprocessing.yaml
        2. Проверка данных (соответствие заданным критериям) - check_data.yaml
        3. Разметка данных автоматизированным алгоритмом разметки данных - auto_marking.yaml
        4. Оценка погрешности работы алгоритма автоматической разметки относительно ручной разметки - error_am.yaml
        5. Извлечение вектора признаков c использованием ручной или автоматической разметки данных  - fe.yaml
        6. Расчет статистик по признакам, статестичесие данные по датасету - statistic.yaml
        7. Проведение экспериментов с классическим ML - ml.yaml
            * Расширенный набор метрик на заданном векторе признаков и гиперпараметрах алгоритмов классификации и регрессии:
                - Бинарная классификаци binary_clf.yaml
                - Многоклассовая классификаци multi_clf.yaml
                - Задача регрессии regr.yaml
            * Отбор признаков:
                - feature_selection.yaml
            * Подбор гиперпараметров алгоритма:
                - optuna_clf.yaml
                - optuna_rgr.yaml

Задачи в процессе интеграции:
1. AutoML
2. Расчет нормировочного коэффициента для амплитуд FT, OC
3. Генерация синтетических данных
4. Анализ выбросов
5. Нейронная сеть MLP
6. Нейронная сеть для анализа сырых сигналов
5. Интеграция анализа тремора
6. Интеграция анализа рук по 2D

Статьи по теме исследования:

