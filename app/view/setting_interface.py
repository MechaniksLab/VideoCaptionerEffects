import webbrowser

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QColor
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget
from qfluentwidgets import ComboBoxSettingCard, CustomColorSettingCard, ExpandLayout
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    HyperlinkCard,
    InfoBar,
    MessageBox,
    OptionsSettingCard,
    PrimaryPushSettingCard,
    PushSettingCard,
    RangeSettingCard,
    ScrollArea,
    SettingCardGroup,
    SwitchSettingCard,
)

from app.common.config import cfg
from app.common.theme_manager import apply_vscode_theme, get_theme_palette
from app.common.signal_bus import signalBus
from app.components.EditComboBoxSettingCard import EditComboBoxSettingCard
from app.components.LineEditSettingCard import LineEditSettingCard
from app.config import APP_NAME, AUTHOR, FEEDBACK_URL, HELP_URL, RELEASE_URL, VERSION, YEAR
from app.core.entities import LLMServiceEnum, TranscribeModelEnum, TranslatorServiceEnum
from app.core.utils.test_opanai import get_openai_models, test_openai
from app.thread.version_manager_thread import VersionManager
from app.components.MySettingCard import ComboBoxSettingCard as MyComboBoxSettingCard
from app.components.MySettingCard import ColorSettingCard


class SettingInterface(ScrollArea):
    """设置界面"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Настройки")
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)
        self.settingLabel = QLabel("Настройки", self)

        # 初始化所有设置组
        self.__initGroups()
        # 初始化所有配置卡片
        self.__initCards()
        # 初始化界面
        self.__initWidget()
        # 初始化布局
        self.__initLayout()
        # 连接信号和槽
        self.__connectSignalToSlot()

    def __initGroups(self):
        """初始化所有设置组"""
        # 转录配置组
        self.transcribeGroup = SettingCardGroup("Параметры распознавания", self.scrollWidget)
        # LLM配置组
        self.llmGroup = SettingCardGroup("Параметры LLM", self.scrollWidget)
        # 翻译服务组
        self.translate_serviceGroup = SettingCardGroup(
            "Сервис перевода", self.scrollWidget
        )
        # 翻译与优化组
        self.translateGroup = SettingCardGroup("Перевод и оптимизация", self.scrollWidget)
        # 字幕合成配置组
        self.subtitleGroup = SettingCardGroup(
            "Параметры синтеза субтитров", self.scrollWidget
        )
        # 保存配置组
        self.saveGroup = SettingCardGroup("Сохранение", self.scrollWidget)
        # 个性化组
        self.personalGroup = SettingCardGroup("Персонализация", self.scrollWidget)
        # 关于组
        self.aboutGroup = SettingCardGroup("О программе", self.scrollWidget)

    def __initCards(self):
        """初始化所有配置卡片"""
        # 转录配置卡片
        self.transcribeModelCard = ComboBoxSettingCard(
            cfg.transcribe_model,
            FIF.MICROPHONE,
            "Модель распознавания",
            "Модель ASR для преобразования речи в текст",
            texts=[model.value for model in cfg.transcribe_model.validator.options],
            parent=self.transcribeGroup,
        )

        # LLM配置卡片
        self.__createLLMServiceCards()

        # 翻译配置卡片
        self.__createTranslateServiceCards()

        # 翻译与优化配置卡片
        self.subtitleCorrectCard = SwitchSettingCard(
            FIF.EDIT,
            "Коррекция субтитров",
            "Выполнять коррекцию сгенерированных субтитров",
            cfg.need_optimize,
            self.translateGroup,
        )
        self.subtitleTranslateCard = SwitchSettingCard(
            FIF.LANGUAGE,
            "Перевод субтитров",
            "Выполнять перевод субтитров в процессе обработки",
            cfg.need_translate,
            self.translateGroup,
        )
        self.targetLanguageCard = ComboBoxSettingCard(
            cfg.target_language,
            FIF.LANGUAGE,
            "Целевой язык",
            "Выберите язык перевода субтитров",
            texts=[lang.value for lang in cfg.target_language.validator.options],
            parent=self.translateGroup,
        )

        # 字幕合成配置卡片
        self.subtitleStyleCard = HyperlinkCard(
            "",
            "Изменить",
            FIF.FONT,
            "Стиль субтитров",
            "Выбор стиля субтитров (цвет, размер, шрифт и т.д.)",
            self.subtitleGroup,
        )
        self.subtitleLayoutCard = HyperlinkCard(
            "",
            "Изменить",
            FIF.FONT,
            "Расположение субтитров",
            "Выбор макета субтитров (одноязычные/двуязычные)",
            self.subtitleGroup,
        )
        self.needVideoCard = SwitchSettingCard(
            FIF.VIDEO,
            "Синтезировать видео",
            "Если включено — создавать видео, если выключено — пропускать",
            cfg.need_video,
            self.subtitleGroup,
        )
        self.softSubtitleCard = SwitchSettingCard(
            FIF.FONT,
            "Мягкие субтитры",
            "Включено: субтитры можно отключать в плеере. Выключено: субтитры вшиваются в кадр",
            cfg.soft_subtitle,
            self.subtitleGroup,
        )

        # 保存配置卡片
        self.savePathCard = PushSettingCard(
            "Рабочая папка",
            FIF.SAVE,
            "Путь к рабочей директории",
            cfg.get(cfg.work_dir),
            self.saveGroup,
        )

        # 个性化配置卡片
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            "Тема приложения",
            "Изменение внешнего вида приложения",
            texts=["Светлая", "Тёмная", "Как в системе"],
            parent=self.personalGroup,
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            "Акцентный цвет",
            "Изменение акцентного цвета приложения",
            self.personalGroup,
        )
        self.uiWindowBgCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_window_bg, "#1E1E1E"),
            FIF.BRUSH,
            "Фон окна",
            "Основной фон страницы/рабочей области",
            self.personalGroup,
        )
        self.uiPanelBgCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_panel_bg, "#252526"),
            FIF.PALETTE,
            "Фон панелей",
            "Цвет фона для панелей и областей ввода",
            self.personalGroup,
        )
        self.uiCardBgCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_card_bg, "#2D2D30"),
            FIF.PALETTE,
            "Фон карточек",
            "Цвет карточек и блоков настроек",
            self.personalGroup,
        )
        self.uiBorderColorCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_border_color, "#3C3C3C"),
            FIF.BRUSH,
            "Цвет границ",
            "Границы таблиц, карточек и полей",
            self.personalGroup,
        )
        self.uiTextColorCard = ColorSettingCard(
            self._cfg_color_or_default(cfg.ui_text_color, "#D4D4D4"),
            FIF.FONT,
            "Основной цвет текста",
            "Базовый цвет текста интерфейса",
            self.personalGroup,
        )
        self.applyThemeCard = PushSettingCard(
            "Применить",
            FIF.BRUSH,
            "Применить тему и цвета",
            "Применяет выбранные цвета и тему без перезапуска",
            self.personalGroup,
        )
        self.zoomCard = OptionsSettingCard(
            cfg.dpiScale,
            FIF.ZOOM,
            "Масштаб интерфейса",
            "Изменение размера виджетов и шрифтов",
            texts=["100%", "125%", "150%", "175%", "200%", "Как в системе"],
            parent=self.personalGroup,
        )
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            "Язык",
            "Выберите язык интерфейса",
            texts=[
                "Китайский (упрощённый)",
                "Китайский (традиционный)",
                "English",
                "Как в системе",
            ],
            parent=self.personalGroup,
        )

        # 关于卡片
        self.helpCard = HyperlinkCard(
            HELP_URL,
            "Открыть страницу помощи",
            FIF.HELP,
            "Помощь",
            f"Новые функции и советы по использованию {APP_NAME}",
            self.aboutGroup,
        )
        self.feedbackCard = PrimaryPushSettingCard(
            "Оставить отзыв",
            FIF.FEEDBACK,
            "Оставить отзыв",
            f"Ваш отзыв помогает улучшать {APP_NAME}",
            self.aboutGroup,
        )
        self.aboutCard = PrimaryPushSettingCard(
            "Проверить обновления",
            FIF.INFO,
            "О программе",
            "© "
            + "Все права защищены"
            + f" {YEAR}, {AUTHOR}. "
            + "Версия"
            + " "
            + VERSION,
            self.aboutGroup,
        )

        # 添加卡片到对应的组
        self.translateGroup.addSettingCard(self.subtitleCorrectCard)
        self.translateGroup.addSettingCard(self.subtitleTranslateCard)
        self.translateGroup.addSettingCard(self.targetLanguageCard)

        self.subtitleGroup.addSettingCard(self.subtitleStyleCard)
        self.subtitleGroup.addSettingCard(self.subtitleLayoutCard)
        self.subtitleGroup.addSettingCard(self.needVideoCard)
        self.subtitleGroup.addSettingCard(self.softSubtitleCard)

        self.saveGroup.addSettingCard(self.savePathCard)

        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.uiWindowBgCard)
        self.personalGroup.addSettingCard(self.uiPanelBgCard)
        self.personalGroup.addSettingCard(self.uiCardBgCard)
        self.personalGroup.addSettingCard(self.uiBorderColorCard)
        self.personalGroup.addSettingCard(self.uiTextColorCard)
        self.personalGroup.addSettingCard(self.applyThemeCard)
        self.personalGroup.addSettingCard(self.zoomCard)
        self.personalGroup.addSettingCard(self.languageCard)

        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.feedbackCard)
        self.aboutGroup.addSettingCard(self.aboutCard)

    def __createLLMServiceCards(self):
        """创建LLM服务相关的配置卡片"""
        # 服务选择卡片
        self.llmServiceCard = ComboBoxSettingCard(
            cfg.llm_service,
            FIF.ROBOT,
            "Сервис LLM",
            "Выберите сервис LLM для сегментации, оптимизации и перевода субтитров",
            texts=[service.value for service in cfg.llm_service.validator.options],
            parent=self.llmGroup,
        )

        # 创建OPENAI官方API链接卡片
        self.openaiOfficialApiCard = HyperlinkCard(
            "https://api.videocaptioner.cn/register?aff=UrLB",
            "Перейти",
            FIF.DEVELOPER_TOOLS,
            f"Официальный API {APP_NAME}",
            "Интеграция нескольких LLM, поддержка быстрой оптимизации и перевода",
            self.llmGroup,
        )
        # 默认隐藏
        self.openaiOfficialApiCard.setVisible(False)

        # 定义每个服务的配置
        service_configs = {
            LLMServiceEnum.OPENAI: {
                "prefix": "openai",
                "api_key_cfg": cfg.openai_api_key,
                "api_base_cfg": cfg.openai_api_base,
                "model_cfg": cfg.openai_model,
                "default_base": "https://api.openai.com/v1",
                "default_models": [
                    "gpt-4o-mini",
                    "gpt-4o",
                    "claude-3-5-sonnet-20241022",
                ],
            },
            LLMServiceEnum.SILICON_CLOUD: {
                "prefix": "silicon_cloud",
                "api_key_cfg": cfg.silicon_cloud_api_key,
                "api_base_cfg": cfg.silicon_cloud_api_base,
                "model_cfg": cfg.silicon_cloud_model,
                "default_base": "https://api.siliconflow.cn/v1",
                "default_models": ["deepseek-ai/DeepSeek-V3"],
            },
            LLMServiceEnum.DEEPSEEK: {
                "prefix": "deepseek",
                "api_key_cfg": cfg.deepseek_api_key,
                "api_base_cfg": cfg.deepseek_api_base,
                "model_cfg": cfg.deepseek_model,
                "default_base": "https://api.deepseek.com/v1",
                "default_models": ["deepseek-chat"],
            },
            LLMServiceEnum.OLLAMA: {
                "prefix": "ollama",
                "api_key_cfg": cfg.ollama_api_key,
                "api_base_cfg": cfg.ollama_api_base,
                "model_cfg": cfg.ollama_model,
                "default_base": "http://localhost:11434/v1",
                "default_models": ["qwen2.5:7b"],
            },
            LLMServiceEnum.LM_STUDIO: {
                "prefix": "LM Studio",
                "api_key_cfg": cfg.lm_studio_api_key,
                "api_base_cfg": cfg.lm_studio_api_base,
                "model_cfg": cfg.lm_studio_model,
                "default_base": "http://localhost:1234/v1",
                "default_models": ["qwen2.5:7b"],
            },
            LLMServiceEnum.GEMINI: {
                "prefix": "gemini",
                "api_key_cfg": cfg.gemini_api_key,
                "api_base_cfg": cfg.gemini_api_base,
                "model_cfg": cfg.gemini_model,
                "default_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "default_models": ["gemini-2.0-flash-exp"],
            },
            LLMServiceEnum.CHATGLM: {
                "prefix": "chatglm",
                "api_key_cfg": cfg.chatglm_api_key,
                "api_base_cfg": cfg.chatglm_api_base,
                "model_cfg": cfg.chatglm_model,
                "default_base": "https://open.bigmodel.cn/api/paas/v4",
                "default_models": ["glm-4-flash"],
            },
            LLMServiceEnum.PUBLIC: {
                "prefix": "public",
                "api_key_cfg": cfg.public_api_key,
                "api_base_cfg": cfg.public_api_base,
                "model_cfg": cfg.public_model,
                "default_base": "https://api.public-model.com/v1",
                "default_models": ["public-model"],
            },
        }

        # 创建服务配置映射
        self.llm_service_configs = {}

        # 为每个服务创建配置卡片
        for service, config in service_configs.items():
            prefix = config["prefix"]

            # 如果是公益模型，只添加配置不创建卡片
            if service == LLMServiceEnum.PUBLIC:
                self.llm_service_configs[service] = {
                    "cards": [],
                    "api_base": None,
                    "api_key": None,
                    "model": None,
                }
                continue

            # 创建API Key卡片
            api_key_card = LineEditSettingCard(
                config["api_key_cfg"],
                FIF.FINGERPRINT,
                "API Key",
                f"Введите API Key для {service.value}",
                "sk-" if service != LLMServiceEnum.OLLAMA else "",
                self.llmGroup,
            )
            setattr(self, f"{prefix}_api_key_card", api_key_card)

            # 创建Base URL卡片
            api_base_card = LineEditSettingCard(
                config["api_base_cfg"],
                FIF.LINK,
                "Base URL",
                f"Введите Base URL для {service.value} (должен содержать /v1)",
                config["default_base"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_api_base_card", api_base_card)

            # 创建模型选择卡片
            model_card = EditComboBoxSettingCard(
                config["model_cfg"],
                FIF.ROBOT,
                "Модель",
                f"Выберите модель {service.value}",
                config["default_models"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_model_card", model_card)

            # 存储服务配置
            cards = [api_key_card, api_base_card, model_card]

            self.llm_service_configs[service] = {
                "cards": cards,
                "api_base": api_base_card,
                "api_key": api_key_card,
                "model": model_card,
            }

        # 创建检查连接卡片
        self.checkLLMConnectionCard = PushSettingCard(
            "Проверить соединение",
            FIF.LINK,
            "Проверка соединения LLM",
            "Проверить доступность API и получить список моделей",
            self.llmGroup,
        )

        # 初始化显示状态
        self.__onLLMServiceChanged(self.llmServiceCard.comboBox.currentText())

    def __createTranslateServiceCards(self):
        """创建翻译服务相关的配置卡片"""
        # 翻译服务选择卡片
        self.translatorServiceCard = ComboBoxSettingCard(
            cfg.translator_service,
            FIF.ROBOT,
            "Сервис перевода",
            "Выберите сервис перевода",
            texts=[
                service.value for service in cfg.translator_service.validator.options
            ],
            parent=self.translate_serviceGroup,
        )

        # 反思翻译开关
        self.needReflectTranslateCard = SwitchSettingCard(
            FIF.EDIT,
            "Рефлексивный перевод",
            "Улучшает качество перевода, но требует больше времени и токенов",
            cfg.need_reflect_translate,
            self.translate_serviceGroup,
        )

        # DeepLx端点配置
        self.deeplxEndpointCard = LineEditSettingCard(
            cfg.deeplx_endpoint,
            FIF.LINK,
            "Бэкенд DeepLx",
            "Введите адрес DeepLx (обязательно при использовании deeplx)",
            "https://api.deeplx.org/translate",
            self.translate_serviceGroup,
        )

        # 批处理大小配置
        self.batchSizeCard = RangeSettingCard(
            cfg.batch_size,
            FIF.ALIGNMENT,
            "Размер пакета",
            "Количество субтитров в одном пакете (рекомендуется кратно 10)",
            parent=self.translate_serviceGroup,
        )

        # 线程数配置
        self.threadNumCard = RangeSettingCard(
            cfg.thread_num,
            FIF.SPEED_HIGH,
            "Количество потоков",
            "Число параллельных запросов: чем больше (в рамках лимитов), тем выше скорость",
            parent=self.translate_serviceGroup,
        )

        # 添加卡片到翻译服务组
        self.translate_serviceGroup.addSettingCard(self.translatorServiceCard)
        self.translate_serviceGroup.addSettingCard(self.needReflectTranslateCard)
        self.translate_serviceGroup.addSettingCard(self.deeplxEndpointCard)
        self.translate_serviceGroup.addSettingCard(self.batchSizeCard)
        self.translate_serviceGroup.addSettingCard(self.threadNumCard)

        # 初始化显示状态
        self.__onTranslatorServiceChanged(
            self.translatorServiceCard.comboBox.currentText()
        )

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName("settingInterface")

        # 初始化样式表
        self.scrollWidget.setObjectName("scrollWidget")
        self.settingLabel.setObjectName("settingLabel")

        # 初始化翻译服务配置卡片的显示状态
        self.__onTranslatorServiceChanged(
            self.translatorServiceCard.comboBox.currentText()
        )

        self.setStyleSheet(
            """        
            SettingInterface, #scrollWidget {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLabel#settingLabel {
                font: 33px 'Microsoft YaHei';
                background-color: transparent;
                color: #D4D4D4;
            }
        """
        )
        self.refresh_theme()

    def __initLayout(self):
        """初始化布局"""
        self.settingLabel.move(36, 30)

        # 添加转录配置卡片
        self.transcribeGroup.addSettingCard(self.transcribeModelCard)

        # 添加LLM配置卡片
        self.llmGroup.addSettingCard(self.llmServiceCard)
        # 添加OPENAI官方API链接卡片
        self.llmGroup.addSettingCard(self.openaiOfficialApiCard)
        for config in self.llm_service_configs.values():
            for card in config["cards"]:
                self.llmGroup.addSettingCard(card)
        self.llmGroup.addSettingCard(self.checkLLMConnectionCard)

        # 将所有组添加到布局
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.transcribeGroup)
        self.expandLayout.addWidget(self.llmGroup)
        self.expandLayout.addWidget(self.translate_serviceGroup)
        self.expandLayout.addWidget(self.translateGroup)
        self.expandLayout.addWidget(self.subtitleGroup)
        self.expandLayout.addWidget(self.saveGroup)
        self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.aboutGroup)

    def __connectSignalToSlot(self):
        """连接信号与槽"""
        cfg.appRestartSig.connect(self.__showRestartTooltip)

        # LLM服务切换
        self.llmServiceCard.comboBox.currentTextChanged.connect(
            self.__onLLMServiceChanged
        )

        # 翻译服务切换
        self.translatorServiceCard.comboBox.currentTextChanged.connect(
            self.__onTranslatorServiceChanged
        )

        # 检查 LLM 连接
        self.checkLLMConnectionCard.clicked.connect(self.checkLLMConnection)

        # 保存路径
        self.savePathCard.clicked.connect(self.__onsavePathCardClicked)

        # 字幕样式修改跳转
        self.subtitleStyleCard.linkButton.clicked.connect(
            lambda: self.window().switchTo(self.window().subtitleStyleInterface)
        )
        self.subtitleLayoutCard.linkButton.clicked.connect(
            lambda: self.window().switchTo(self.window().subtitleStyleInterface)
        )

        # 个性化
        self.themeCard.optionChanged.connect(lambda _ci: self._on_theme_mode_changed())
        self.applyThemeCard.clicked.connect(self._apply_theme_from_controls)

        # 反馈
        self.feedbackCard.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL))
        )

        # 关于
        self.aboutCard.clicked.connect(self.checkUpdate)

        # 全局 signalBus
        self.transcribeModelCard.comboBox.currentTextChanged.connect(
            signalBus.transcription_model_changed
        )
        self.subtitleCorrectCard.checkedChanged.connect(
            signalBus.subtitle_optimization_changed
        )
        self.subtitleTranslateCard.checkedChanged.connect(
            signalBus.subtitle_translation_changed
        )
        self.targetLanguageCard.comboBox.currentTextChanged.connect(
            signalBus.target_language_changed
        )
        self.softSubtitleCard.checkedChanged.connect(signalBus.soft_subtitle_changed)
        self.needVideoCard.checkedChanged.connect(signalBus.need_video_changed)

    def _cfg_color_or_default(self, item, fallback: str) -> QColor:
        value = cfg.get(item)
        q = QColor(str(value) if value is not None else "")
        if not q.isValid():
            q = QColor(fallback)
        return q

    def _on_theme_mode_changed(self):
        palette = get_theme_palette()
        if cfg.get(cfg.themeMode).name == "LIGHT":
            self.uiWindowBgCard.setColor(QColor("#F3F3F3"))
            self.uiPanelBgCard.setColor(QColor("#FFFFFF"))
            self.uiCardBgCard.setColor(QColor("#FFFFFF"))
            self.uiBorderColorCard.setColor(QColor("#E1E1E1"))
            self.uiTextColorCard.setColor(QColor("#1F1F1F"))
        elif cfg.get(cfg.themeMode).name == "DARK":
            self.uiWindowBgCard.setColor(QColor("#1E1E1E"))
            self.uiPanelBgCard.setColor(QColor("#252526"))
            self.uiCardBgCard.setColor(QColor("#2D2D30"))
            self.uiBorderColorCard.setColor(QColor("#3C3C3C"))
            self.uiTextColorCard.setColor(QColor("#D4D4D4"))
        else:
            self.uiWindowBgCard.setColor(QColor(palette["window_bg"]))
            self.uiPanelBgCard.setColor(QColor(palette["panel_bg"]))
            self.uiCardBgCard.setColor(QColor(palette["card_bg"]))
            self.uiBorderColorCard.setColor(QColor(palette["border"]))
            self.uiTextColorCard.setColor(QColor(palette["text"]))

    def _apply_theme_from_controls(self):
        cfg.set(cfg.ui_window_bg, self.uiWindowBgCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_panel_bg, self.uiPanelBgCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_card_bg, self.uiCardBgCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_border_color, self.uiBorderColorCard.colorPicker.color.name(QColor.HexRgb))
        cfg.set(cfg.ui_text_color, self.uiTextColorCard.colorPicker.color.name(QColor.HexRgb))

        apply_vscode_theme(refresh_widgets=True)
        self.refresh_theme()
        InfoBar.success(
            "Тема применена",
            "Новые цвета интерфейса применены в реальном времени",
            duration=2200,
            parent=self,
        )

    def refresh_theme(self):
        p = get_theme_palette()
        self.settingLabel.setStyleSheet(
            f"font: 33px 'Microsoft YaHei'; background: transparent; color: {p['text']};"
        )

    def __showRestartTooltip(self):
        """显示重启提示"""
        InfoBar.success(
            "Успешно",
            "Настройки вступят в силу после перезапуска",
            duration=1500,
            parent=self,
        )

    def __onsavePathCardClicked(self):
        """处理保存路径卡片点击事件"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку", "./")
        if not folder or cfg.get(cfg.work_dir) == folder:
            return
        cfg.set(cfg.work_dir, folder)
        self.savePathCard.setContent(folder)

    def checkLLMConnection(self):
        """检查 LLM 连接"""
        # 获取当前选中的服务
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())

        # 获取服务配置
        service_config = self.llm_service_configs.get(current_service)
        if not service_config:
            return

        # 如果是公益模型，使用配置文件中的值
        if current_service == LLMServiceEnum.PUBLIC:
            api_base = cfg.public_api_base.value
            api_key = cfg.public_api_key.value
            model = cfg.public_model.value
        else:
            api_base = (
                service_config["api_base"].lineEdit.text()
                if service_config["api_base"]
                else ""
            )
            api_key = (
                service_config["api_key"].lineEdit.text()
                if service_config["api_key"]
                else ""
            )
            model = (
                service_config["model"].comboBox.currentText()
                if service_config["model"]
                else ""
            )

        # 检查 API Base 是否属于网址
        if not api_base.startswith("http"):
            InfoBar.error(
                "Ошибка",
                "Введите корректный API Base (должен содержать /v1)",
                duration=3000,
                parent=self,
            )
            return

        # 禁用检查按钮，显示加载状态
        self.checkLLMConnectionCard.button.setEnabled(False)
        self.checkLLMConnectionCard.button.setText("Проверка...")

        # 创建并启动线程
        self.connection_thread = LLMConnectionThread(api_base, api_key, model)
        self.connection_thread.finished.connect(self.onConnectionCheckFinished)
        self.connection_thread.error.connect(self.onConnectionCheckError)
        self.connection_thread.start()

    def onConnectionCheckError(self, message):
        """处理连接检查错误事件"""
        self.checkLLMConnectionCard.button.setEnabled(True)
        self.checkLLMConnectionCard.button.setText("Проверить соединение")
        InfoBar.error("Ошибка проверки LLM", message, duration=3000, parent=self)

    def onConnectionCheckFinished(self, is_success, message, models):
        """处理连接检查完成事件"""
        self.checkLLMConnectionCard.button.setEnabled(True)
        self.checkLLMConnectionCard.button.setText("Проверить соединение")

        # 获取当前服务
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())

        if models:
            # 更新当前服务的模型列表
            service_config = self.llm_service_configs.get(current_service)
            if service_config and service_config["model"]:
                temp = service_config["model"].comboBox.currentText()
                service_config["model"].setItems(models)
                service_config["model"].comboBox.setCurrentText(temp)

            InfoBar.success(
                "Список моделей получен",
                "Всего моделей: " + str(len(models)),
                duration=3000,
                parent=self,
            )
        if not is_success:
            InfoBar.error(
                "Ошибка проверки LLM", message, duration=3000, parent=self
            )
        else:
            InfoBar.success(
                "Проверка LLM успешна", message, duration=3000, parent=self
            )

    def checkUpdate(self):
        webbrowser.open(RELEASE_URL)

    def __onLLMServiceChanged(self, service):
        """处理LLM服务切换事件"""
        current_service = LLMServiceEnum(service)

        # 隐藏所有卡片
        for config in self.llm_service_configs.values():
            for card in config["cards"]:
                card.setVisible(False)

        # 隐藏OPENAI官方API链接卡片
        self.openaiOfficialApiCard.setVisible(False)

        # 显示选中服务的卡片
        if current_service in self.llm_service_configs:
            for card in self.llm_service_configs[current_service]["cards"]:
                card.setVisible(True)

            # 为OLLAMA和LM_STUDIO设置默认API Key
            service_config = self.llm_service_configs[current_service]
            if current_service == LLMServiceEnum.OLLAMA and service_config["api_key"]:
                # 如果API Key为空，设置默认值"ollama"
                if not service_config["api_key"].lineEdit.text():
                    service_config["api_key"].lineEdit.setText("ollama")
            if (
                current_service == LLMServiceEnum.LM_STUDIO
                and service_config["api_key"]
            ):
                # 如果API Key为空，设置默认值 "lm-studio"
                if not service_config["api_key"].lineEdit.text():
                    service_config["api_key"].lineEdit.setText("lm-studio")

            # 如果是OPENAI服务，显示官方API链接卡片
            if current_service == LLMServiceEnum.OPENAI:
                self.openaiOfficialApiCard.setVisible(True)

        # 更新布局
        self.llmGroup.adjustSize()
        self.expandLayout.update()

    def __onTranslatorServiceChanged(self, service):
        openai_cards = [
            self.needReflectTranslateCard,
            self.batchSizeCard,
        ]
        deeplx_cards = [self.deeplxEndpointCard]

        all_cards = openai_cards + deeplx_cards
        for card in all_cards:
            card.setVisible(False)

        # 根据选择的服务显示相应的配置卡片
        if service in [TranslatorServiceEnum.DEEPLX.value]:
            for card in deeplx_cards:
                card.setVisible(True)
        elif service in [TranslatorServiceEnum.OPENAI.value]:
            for card in openai_cards:
                card.setVisible(True)

        # 更新布局
        self.translate_serviceGroup.adjustSize()
        self.expandLayout.update()


class LLMConnectionThread(QThread):
    finished = pyqtSignal(bool, str, list)
    error = pyqtSignal(str)

    def __init__(self, api_base, api_key, model):
        super().__init__()
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def run(self):
        """检查 LLM 连接并获取模型列表"""
        try:
            is_success, message = test_openai(self.api_base, self.api_key, self.model)
            models = get_openai_models(self.api_base, self.api_key)
            self.finished.emit(is_success, message, models)
        except Exception as e:
            self.error.emit(str(e))
