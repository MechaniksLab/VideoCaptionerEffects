import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFontDatabase
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CardWidget
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import (
    ImageLabel,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    MessageBoxBase,
    PushSettingCard,
    ScrollArea,
    SettingCardGroup,
    Slider,
    PushButton,
)

from app.common.config import cfg
from app.common.signal_bus import signalBus
from app.components.MySettingCard import (
    ColorSettingCard,
    ComboBoxSettingCard,
    DoubleSpinBoxSettingCard,
    SpinBoxSettingCard,
)
from app.config import SUBTITLE_STYLE_PATH, ASSETS_PATH
from app.core.entities import SplitTypeEnum
from app.core.subtitle_processor.effect_manager import EffectManager
from app.core.utils.subtitle_preview import generate_preview, generate_preview_video

LAYOUT_LABEL_TO_VALUE = {
    "Перевод сверху": "译文在上",
    "Оригинал сверху": "原文在上",
    "Только перевод": "仅译文",
    "Только оригинал": "仅原文",
}
LAYOUT_VALUE_TO_LABEL = {v: k for k, v in LAYOUT_LABEL_TO_VALUE.items()}

ORIENTATION_LABEL_TO_VALUE = {
    "Горизонтальный": "横屏",
    "Вертикальный": "竖屏",
}
ORIENTATION_VALUE_TO_LABEL = {v: k for k, v in ORIENTATION_LABEL_TO_VALUE.items()}

STYLE_PRESET_LABEL_TO_VALUE = {
    "Свой (Custom)": "custom",
    "TikTok Dynamic": "tiktok_dynamic",
    "YouTube Shorts Clean": "shorts_clean",
    "Minimal Classic": "minimal_classic",
    "Karaoke Pro": "karaoke_pro",
    "Cinema Gradient": "cinema_gradient",
    "Neon Pulse": "neon_pulse",
}
STYLE_PRESET_VALUE_TO_LABEL = {v: k for k, v in STYLE_PRESET_LABEL_TO_VALUE.items()}

TOGGLE_LABEL_TO_VALUE = {
    "Выкл": False,
    "Вкл": True,
}
TOGGLE_VALUE_TO_LABEL = {v: k for k, v in TOGGLE_LABEL_TO_VALUE.items()}

GRADIENT_MODE_LABEL_TO_VALUE = {
    "Без градиента": "off",
    "2 цвета": "two_color",
    "Радужный": "rainbow",
}
GRADIENT_MODE_VALUE_TO_LABEL = {v: k for k, v in GRADIENT_MODE_LABEL_TO_VALUE.items()}

SPLIT_TYPE_LABEL_TO_VALUE = {
    "По предложениям": SplitTypeEnum.SENTENCE,
    "По словам": SplitTypeEnum.SEMANTIC,
}
SPLIT_TYPE_VALUE_TO_LABEL = {
    SplitTypeEnum.SENTENCE: "По предложениям",
    SplitTypeEnum.SEMANTIC: "По словам",
    SplitTypeEnum.SENTENCE.value: "По предложениям",
    SplitTypeEnum.SEMANTIC.value: "По словам",
}

MOTION_DIRECTION_LABEL_TO_VALUE = {
    "Снизу вверх": "up",
    "Сверху вниз": "down",
    "Слева направо": "left",
    "Справа налево": "right",
}
MOTION_DIRECTION_VALUE_TO_LABEL = {
    v: k for k, v in MOTION_DIRECTION_LABEL_TO_VALUE.items()
}

MOTION_EASING_LABEL_TO_VALUE = {
    "Ease Out (мягкий финиш)": "ease_out",
    "Ease In (мягкий старт)": "ease_in",
    "Ease In-Out": "ease_in_out",
    "Linear": "linear",
}
MOTION_EASING_VALUE_TO_LABEL = {v: k for k, v in MOTION_EASING_LABEL_TO_VALUE.items()}

SENTENCE_MODE_EFFECTS = {
    "none",
    "fade_in",
    "fade_out",
    "fade_in_out",
    "typewriter",
    "word_highlight",
    "shine",
}

WORD_MODE_EFFECTS = {
    "bounce",
    "pulse",
    "wave",
    "spin",
    "zoom_in",
    "swing",
    "slide_up",
    "slide_left",
    "pop_rotate",
    "shake",
    "neon_flicker",
    "glitch",
    "twinkle",
    "rainbow",
}

PREVIEW_TEXTS_SENTENCE = {
    "Длинный текст": (
        "This is a long text used for testing subtitle preview and style settings.",
        "这是一段用于测试字幕预览和样式设置的长文本内容",
    ),
    "Средний текст": (
        "Welcome to apply for the prestigious South China Normal University!",
        "欢迎报考百年名校华南师范大学",
    ),
    "Короткий текст": ("Elementary school students know this", "小学二年级的都知道"),
}

PREVIEW_TEXTS_WORD = {
    "По словам — коротко": (
        "Word by word karaoke preview",
        "Слово за словом караоке",
    ),
    "По словам — ритм": (
        "one two three four five",
        "раз два три четыре пять",
    ),
    "По словам — акцент": (
        "highlight every word clearly",
        "подсветка каждого слова",
    ),
}

DEFAULT_BG_LANDSCAPE = {
    "path": ASSETS_PATH / "default_bg_landscape.png",
    "width": 1280,
    "height": 720,
}
DEFAULT_BG_PORTRAIT = {
    "path": ASSETS_PATH / "default_bg_portrait.png",
    "width": 480,
    "height": 852,
}


class PreviewThread(QThread):
    previewReady = pyqtSignal(str, int)

    def __init__(
        self,
        style_str: str,
        preview_text: Tuple[str, Optional[str]],
        bg_path: str,
        width: int,
        height: int,
        effect_type: str,
        effect_duration_ms: int,
        effect_intensity: float,
        rainbow_end_color: str,
        motion_direction: str,
        motion_amplitude: float,
        motion_easing: str,
        motion_jitter: float,
        karaoke_mode: bool,
        karaoke_window_ms: int,
        auto_contrast: bool,
        anti_flicker: bool,
        gradient_mode: str,
        gradient_color_1: str,
        gradient_color_2: str,
        preview_time_sec: float,
        request_id: int,
    ):
        """
        Args:
            style_str: ASS 样式字符串
            preview_text: 预览文本元组 (主字幕, 副字幕), 副字幕可选
        """
        super().__init__()
        self.style_str = style_str
        self.preview_text = preview_text
        self.bg_path = bg_path
        self.width = width
        self.height = height
        self.effect_type = effect_type
        self.effect_duration_ms = effect_duration_ms
        self.effect_intensity = effect_intensity
        self.rainbow_end_color = rainbow_end_color
        self.motion_direction = motion_direction
        self.motion_amplitude = motion_amplitude
        self.motion_easing = motion_easing
        self.motion_jitter = motion_jitter
        self.karaoke_mode = karaoke_mode
        self.karaoke_window_ms = karaoke_window_ms
        self.auto_contrast = auto_contrast
        self.anti_flicker = anti_flicker
        self.gradient_mode = gradient_mode
        self.gradient_color_1 = gradient_color_1
        self.gradient_color_2 = gradient_color_2
        self.preview_time_sec = preview_time_sec
        self.request_id = request_id

    def run(self):
        frame_token = f"{self.request_id}_{int(self.preview_time_sec * 1000)}"
        preview_path = generate_preview(
            style_str=self.style_str,
            preview_text=self.preview_text,
            bg_path=self.bg_path,
            width=self.width,
            height=self.height,
            effect_type=self.effect_type,
            effect_duration_ms=self.effect_duration_ms,
            effect_intensity=self.effect_intensity,
            rainbow_end_color=self.rainbow_end_color,
            motion_direction=self.motion_direction,
            motion_amplitude=self.motion_amplitude,
            motion_easing=self.motion_easing,
            motion_jitter=self.motion_jitter,
            karaoke_mode=self.karaoke_mode,
            karaoke_window_ms=self.karaoke_window_ms,
            auto_contrast=self.auto_contrast,
            anti_flicker=self.anti_flicker,
            gradient_mode=self.gradient_mode,
            gradient_color_1=self.gradient_color_1,
            gradient_color_2=self.gradient_color_2,
            preview_time_sec=self.preview_time_sec,
            frame_token=frame_token,
        )
        self.previewReady.emit(preview_path, self.request_id)


class SubtitleStyleInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("SubtitleStyleInterface")
        self.setWindowTitle("Настройка стиля субтитров")

        # 创建主布局
        self.hBoxLayout = QHBoxLayout(self)

        # 初始化界面组件
        self._initSettingsArea()
        self._initPreviewArea()
        self._initSettingCards()
        self._initLayout()
        self._initStyle()

        # 添加一个标志位来控制是否触发onSettingChanged
        self._loading_style = False

        # Live preview state (должен быть инициализирован до __setValues,
        # т.к. loadStyle() внутри __setValues вызывает updatePreview())
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(40)  # ~25 FPS
        self._live_timer.timeout.connect(self._onLiveTimerTick)
        self._live_playing = False

        self._preview_debounce_timer = QTimer(self)
        self._preview_debounce_timer.setSingleShot(True)
        self._preview_debounce_timer.setInterval(80)
        self._preview_debounce_timer.timeout.connect(self._renderPreviewNow)

        self._preview_thread = None
        self._preview_pending = False
        self._preview_request_id = 0
        self._latest_applied_request_id = -1

        # 设置初始值,加载样式
        self.__setValues()

        # 连接信号
        self.connectSignals()

        self._update_motion_controls_state()

    def _initSettingsArea(self):
        """初始化左侧设置区域"""
        self.settingsScrollArea = ScrollArea()
        self.settingsScrollArea.setFixedWidth(350)
        self.settingsWidget = QWidget()
        self.settingsLayout = QVBoxLayout(self.settingsWidget)
        self.settingsScrollArea.setWidget(self.settingsWidget)
        self.settingsScrollArea.setWidgetResizable(True)

        # 创建设置组
        self.layoutGroup = SettingCardGroup("Расположение и эффекты", self.settingsWidget)
        self.mainGroup = SettingCardGroup("Стиль основного субтитра", self.settingsWidget)
        self.subGroup = SettingCardGroup("Стиль дополнительного субтитра", self.settingsWidget)
        self.previewGroup = SettingCardGroup("Предпросмотр", self.settingsWidget)

    def _initPreviewArea(self):
        """初始化右侧预览区域"""
        self.previewCard = CardWidget()
        self.previewLayout = QVBoxLayout(self.previewCard)
        self.previewLayout.setSpacing(16)

        # 顶部预览区域
        self.previewTopWidget = QWidget()
        self.previewTopWidget.setFixedHeight(430)
        self.previewTopLayout = QVBoxLayout(self.previewTopWidget)

        self.previewLabel = BodyLabel("Предпросмотр")
        self.previewImage = ImageLabel()
        self.previewImage.setAlignment(Qt.AlignCenter)
        self.previewTopLayout.addWidget(self.previewImage, 0, Qt.AlignCenter)
        self.previewTopLayout.setAlignment(Qt.AlignVCenter)

        # 底部控件区域
        self.previewBottomWidget = QWidget()
        self.previewBottomLayout = QVBoxLayout(self.previewBottomWidget)

        self.timelineWidget = QWidget()
        self.timelineLayout = QHBoxLayout(self.timelineWidget)
        self.timelineLayout.setContentsMargins(0, 0, 0, 0)
        self.timelineLayout.setSpacing(8)

        self.playPreviewButton = PushButton("▶ Live", self.timelineWidget)
        self.playPreviewButton.setFixedWidth(78)
        self.timelineSlider = Slider(Qt.Horizontal, self.timelineWidget)
        self.timelineSlider.setRange(0, 1000)
        self.timelineSlider.setValue(0)
        self.timelineSlider.setSingleStep(40)
        self.timelineLabel = BodyLabel("0.00 c", self.timelineWidget)
        self.timelineLabel.setMinimumWidth(55)

        self.timelineLayout.addWidget(self.playPreviewButton)
        self.timelineLayout.addWidget(self.timelineSlider, 1)
        self.timelineLayout.addWidget(self.timelineLabel)

        self.styleNameComboBox = ComboBoxSettingCard(
            FIF.VIEW, "Выбор стиля", "Выберите сохранённый пресет стиля", texts=[]
        )

        self.newStyleButton = PushSettingCard(
            "Новый стиль",
            FIF.ADD,
            "Новый стиль",
            "Создать новый пресет на основе текущих настроек",
        )

        self.openStyleFolderButton = PushSettingCard(
            "Открыть папку стилей",
            FIF.FOLDER,
            "Открыть папку стилей",
            "Открыть папку стилей в проводнике",
        )

        self.previewEffectButton = PushSettingCard(
            "Показать анимацию",
            FIF.PLAY,
            "Показать анимацию эффекта",
            "Сгенерировать короткое видео предпросмотра и открыть его",
        )

        self.wordTimestampHintLabel = BodyLabel(
            "⚠ Для режима «по словам» и точного karaoke нужны word timestamps из ASR.")
        self.wordTimestampHintLabel.setVisible(False)
        self.wordTimestampHintLabel.setStyleSheet(
            "color: #F5A524; font-size: 12px; padding: 2px 0;"
        )

        self.previewBottomLayout.addWidget(self.styleNameComboBox)
        self.previewBottomLayout.addWidget(self.newStyleButton)
        self.previewBottomLayout.addWidget(self.openStyleFolderButton)
        self.previewBottomLayout.addWidget(self.previewEffectButton)
        self.previewBottomLayout.addWidget(self.wordTimestampHintLabel)
        self.previewBottomLayout.addWidget(self.timelineWidget)

        self.previewLayout.addWidget(self.previewTopWidget)
        self.previewLayout.addWidget(self.previewBottomWidget)
        self.previewLayout.addStretch(1)

    def _initSettingCards(self):
        """初始化所有设置卡片"""
        self.effect_options = EffectManager.get_effect_options()

        # 字幕排布设置
        self.layoutCard = ComboBoxSettingCard(
            FIF.ALIGNMENT,
            "Расположение субтитров",
            "Порядок отображения основного и дополнительного текста",
            texts=list(LAYOUT_LABEL_TO_VALUE.keys()),
        )

        self.effectCard = ComboBoxSettingCard(
            FIF.BRUSH,
            "Эффект субтитров",
            "Выберите анимацию появления субтитров",
            texts=list(self.effect_options.keys()),
        )

        self.effectDurationCard = SpinBoxSettingCard(
            FIF.STOP_WATCH,
            "Длительность эффекта (мс)",
            "Сколько длится анимация появления",
            minimum=50,
            maximum=5000,
        )

        self.effectIntensityCard = SpinBoxSettingCard(
            FIF.ZOOM,
            "Интенсивность эффекта (%)",
            "Сила выраженности анимации",
            minimum=10,
            maximum=300,
        )

        self.rainbowEndColorCard = ColorSettingCard(
            QColor(0, 0, 255),
            FIF.PALETTE,
            "Конечный цвет радуги",
            "Цвет, в который переходит текст при эффекте Радуга",
        )

        self.presetCard = ComboBoxSettingCard(
            FIF.BOOK_SHELF,
            "Пресет стиля",
            "Готовые наборы для шортсов",
            texts=list(STYLE_PRESET_LABEL_TO_VALUE.keys()),
        )

        self.motionDirectionCard = ComboBoxSettingCard(
            FIF.MOVE,
            "Направление анимации",
            "Работает для: Прыжок, Волна, Скольжение, Дрожание, Увеличение",
            texts=list(MOTION_DIRECTION_LABEL_TO_VALUE.keys()),
        )

        self.motionAmplitudeCard = SpinBoxSettingCard(
            FIF.ZOOM,
            "Амплитуда движения (%)",
            "Работает для motion-эффектов (см. выше)",
            minimum=20,
            maximum=300,
        )

        self.motionEasingCard = ComboBoxSettingCard(
            FIF.SPEED_HIGH,
            "Плавность (easing)",
            "Профиль скорости для motion-эффектов",
            texts=list(MOTION_EASING_LABEL_TO_VALUE.keys()),
        )

        self.motionJitterCard = SpinBoxSettingCard(
            FIF.SYNC,
            "Дрожание (jitter, %) ",
            "Микро-смещение позиции (только motion-эффекты)",
            minimum=0,
            maximum=100,
        )

        self.karaokeModeCard = ComboBoxSettingCard(
            FIF.MUSIC,
            "Караоке-режим",
            "Подсветка слов по времени",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.karaokeWindowCard = SpinBoxSettingCard(
            FIF.STOP_WATCH,
            "Окно караоке (мс)",
            "Длительность подсветки слов",
            minimum=200,
            maximum=4000,
        )

        # Единая точка настроек сегментации/пунктуации (перенесено из окна шестерёнки)
        self.needSplitCard = ComboBoxSettingCard(
            FIF.TILES,
            "Умное разбиение (LLM)",
            "Вкл: разбиение на фразы/предложения. Выкл: исходные сегменты",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.splitTypeCard = ComboBoxSettingCard(
            FIF.ALIGNMENT,
            "Тип разбиения",
            "Как делить субтитры при включённом умном разбиении",
            texts=list(SPLIT_TYPE_LABEL_TO_VALUE.keys()),
        )

        self.maxWordCountCjkCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Макс. символов CJK",
            "Лимит длины строки для китайского/японского/корейского",
            minimum=8,
            maximum=100,
        )

        self.maxWordCountEnglishCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Макс. слов (англ.)",
            "Лимит длины строки для английского",
            minimum=8,
            maximum=100,
        )

        self.removePunctuationCard = ComboBoxSettingCard(
            FIF.EDIT,
            "Убирать конечную пунктуацию",
            "Удалять хвостовые знаки препинания в конце строк",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.autoContrastCard = ComboBoxSettingCard(
            FIF.CONSTRACT,
            "Авто-контраст",
            "Добавляет адаптивную читаемость (обводка/тень)",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.antiFlickerCard = ComboBoxSettingCard(
            FIF.SYNC,
            "Анти-мигание",
            "Сглаживает резкие анимационные скачки",
            texts=list(TOGGLE_LABEL_TO_VALUE.keys()),
        )

        self.gradientModeCard = ComboBoxSettingCard(
            FIF.PALETTE,
            "Градиент текста",
            "Режим многоцветной раскраски",
            texts=list(GRADIENT_MODE_LABEL_TO_VALUE.keys()),
        )

        self.gradientColor1Card = ColorSettingCard(
            QColor(255, 255, 255),
            FIF.PALETTE,
            "Градиент: цвет 1",
            "Начальный цвет 2-цветного градиента",
        )

        self.gradientColor2Card = ColorSettingCard(
            QColor(102, 204, 255),
            FIF.PALETTE,
            "Градиент: цвет 2",
            "Конечный цвет 2-цветного градиента",
        )

        # 垂直间距
        self.verticalSpacingCard = SpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Вертикальный отступ",
            "Расстояние от нижнего края экрана",
            minimum=8,
            maximum=10000,
        )

        # 主字幕样式设置
        self.mainFontCard = ComboBoxSettingCard(
            FIF.FONT,
            "Шрифт основного субтитра",
            "Выбор шрифта основного текста",
            texts=["Arial"],
        )

        self.mainSizeCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Размер основного субтитра",
            "Размер шрифта основного текста",
            minimum=8,
            maximum=1000,
        )

        self.mainSpacingCard = DoubleSpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Интервал символов (основной)",
            "Межбуквенное расстояние основного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        self.mainColorCard = ColorSettingCard(
            QColor(255, 255, 255),
            FIF.PALETTE,
            "Цвет основного субтитра",
            "Цвет текста основного субтитра",
        )

        self.mainOutlineColorCard = ColorSettingCard(
            QColor(0, 0, 0),
            FIF.PALETTE,
            "Цвет обводки (основной)",
            "Цвет контура основного текста",
        )

        self.mainOutlineSizeCard = DoubleSpinBoxSettingCard(
            FIF.ZOOM,
            "Толщина обводки (основной)",
            "Толщина контура основного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        self.mainShadowCard = DoubleSpinBoxSettingCard(
            FIF.BRUSH,
            "Тень (основной)",
            "Смещение тени основного текста",
            minimum=0.0,
            maximum=20.0,
            decimals=1,
        )

        self.mainShadowColorCard = ColorSettingCard(
            QColor(0, 0, 0, 160),
            FIF.PALETTE,
            "Цвет тени (основной)",
            "Цвет тени основного текста",
        )

        self.mainBlurCard = DoubleSpinBoxSettingCard(
            FIF.HIGHTLIGHT,
            "Свечение/размытие (основной)",
            "Сила мягкого свечения основного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        # 副字幕样式设置
        self.subFontCard = ComboBoxSettingCard(
            FIF.FONT,
            "Шрифт дополнительного субтитра",
            "Выбор шрифта дополнительного текста",
            texts=["Arial"],
        )

        self.subSizeCard = SpinBoxSettingCard(
            FIF.FONT_SIZE,
            "Размер дополнительного субтитра",
            "Размер шрифта дополнительного текста",
            minimum=8,
            maximum=1000,
        )

        self.subSpacingCard = DoubleSpinBoxSettingCard(
            FIF.ALIGNMENT,
            "Интервал символов (доп.)",
            "Межбуквенное расстояние дополнительного текста",
            minimum=0.0,
            maximum=50.0,
            decimals=1,
        )

        self.subColorCard = ColorSettingCard(
            QColor(255, 255, 255),
            FIF.PALETTE,
            "Цвет дополнительного субтитра",
            "Цвет текста дополнительного субтитра",
        )

        self.subOutlineColorCard = ColorSettingCard(
            QColor(0, 0, 0),
            FIF.PALETTE,
            "Цвет обводки (доп.)",
            "Цвет контура дополнительного текста",
        )

        self.subOutlineSizeCard = DoubleSpinBoxSettingCard(
            FIF.ZOOM,
            "Толщина обводки (доп.)",
            "Толщина контура дополнительного текста",
            minimum=0.0,
            maximum=50.0,
            decimals=1,
        )

        self.subShadowCard = DoubleSpinBoxSettingCard(
            FIF.BRUSH,
            "Тень (доп.)",
            "Смещение тени дополнительного текста",
            minimum=0.0,
            maximum=20.0,
            decimals=1,
        )

        self.subShadowColorCard = ColorSettingCard(
            QColor(0, 0, 0, 160),
            FIF.PALETTE,
            "Цвет тени (доп.)",
            "Цвет тени дополнительного текста",
        )

        self.subBlurCard = DoubleSpinBoxSettingCard(
            FIF.HIGHTLIGHT,
            "Свечение/размытие (доп.)",
            "Сила мягкого свечения дополнительного текста",
            minimum=0.0,
            maximum=10.0,
            decimals=1,
        )

        # 预览设置
        self.previewTextCard = ComboBoxSettingCard(
            FIF.MESSAGE,
            "Текст для предпросмотра",
            "Выберите набор текста для проверки стиля",
            texts=list(PREVIEW_TEXTS_SENTENCE.keys()),
            parent=self.previewGroup,
        )

        self.orientationCard = ComboBoxSettingCard(
            FIF.LAYOUT,
            "Ориентация предпросмотра",
            "Горизонтальный или вертикальный формат",
            texts=list(ORIENTATION_LABEL_TO_VALUE.keys()),
            parent=self.previewGroup,
        )

        self.previewImageCard = PushSettingCard(
            "Выбрать изображение",
            FIF.PHOTO,
            "Фон предпросмотра",
            "Выберите фоновую картинку для предпросмотра",
            parent=self.previewGroup,
        )

    def _initLayout(self):
        """初始化布局"""
        # 添加卡片到组
        self.layoutGroup.addSettingCard(self.splitTypeCard)
        self.layoutGroup.addSettingCard(self.needSplitCard)
        self.layoutGroup.addSettingCard(self.layoutCard)
        self.layoutGroup.addSettingCard(self.effectCard)
        self.layoutGroup.addSettingCard(self.effectDurationCard)
        self.layoutGroup.addSettingCard(self.effectIntensityCard)
        self.layoutGroup.addSettingCard(self.karaokeModeCard)
        self.layoutGroup.addSettingCard(self.karaokeWindowCard)
        self.layoutGroup.addSettingCard(self.presetCard)
        self.layoutGroup.addSettingCard(self.motionDirectionCard)
        self.layoutGroup.addSettingCard(self.motionAmplitudeCard)
        self.layoutGroup.addSettingCard(self.motionEasingCard)
        self.layoutGroup.addSettingCard(self.motionJitterCard)
        self.layoutGroup.addSettingCard(self.rainbowEndColorCard)
        self.layoutGroup.addSettingCard(self.maxWordCountCjkCard)
        self.layoutGroup.addSettingCard(self.maxWordCountEnglishCard)
        self.layoutGroup.addSettingCard(self.removePunctuationCard)
        self.layoutGroup.addSettingCard(self.autoContrastCard)
        self.layoutGroup.addSettingCard(self.antiFlickerCard)
        self.layoutGroup.addSettingCard(self.gradientModeCard)
        self.layoutGroup.addSettingCard(self.gradientColor1Card)
        self.layoutGroup.addSettingCard(self.gradientColor2Card)
        self.layoutGroup.addSettingCard(self.verticalSpacingCard)
        self.mainGroup.addSettingCard(self.mainFontCard)
        self.mainGroup.addSettingCard(self.mainSizeCard)
        self.mainGroup.addSettingCard(self.mainSpacingCard)
        self.mainGroup.addSettingCard(self.mainColorCard)
        self.mainGroup.addSettingCard(self.mainOutlineColorCard)
        self.mainGroup.addSettingCard(self.mainOutlineSizeCard)
        self.mainGroup.addSettingCard(self.mainShadowCard)
        self.mainGroup.addSettingCard(self.mainShadowColorCard)
        self.mainGroup.addSettingCard(self.mainBlurCard)

        self.subGroup.addSettingCard(self.subFontCard)
        self.subGroup.addSettingCard(self.subSizeCard)
        self.subGroup.addSettingCard(self.subSpacingCard)
        self.subGroup.addSettingCard(self.subColorCard)
        self.subGroup.addSettingCard(self.subOutlineColorCard)
        self.subGroup.addSettingCard(self.subOutlineSizeCard)
        self.subGroup.addSettingCard(self.subShadowCard)
        self.subGroup.addSettingCard(self.subShadowColorCard)
        self.subGroup.addSettingCard(self.subBlurCard)

        self.previewGroup.addSettingCard(self.previewTextCard)
        self.previewGroup.addSettingCard(self.orientationCard)
        self.previewGroup.addSettingCard(self.previewImageCard)

        # 添加组到布局
        self.settingsLayout.addWidget(self.layoutGroup)
        self.settingsLayout.addWidget(self.mainGroup)
        self.settingsLayout.addWidget(self.subGroup)
        self.settingsLayout.addWidget(self.previewGroup)
        self.settingsLayout.addStretch(1)

        # 添加左右两侧到主布局
        self.hBoxLayout.addWidget(self.settingsScrollArea)
        self.hBoxLayout.addWidget(self.previewCard)

    def _initStyle(self):
        """初始化样式"""
        self.settingsWidget.setObjectName("settingsWidget")
        self.setStyleSheet(
            """        
            SubtitleStyleInterface, #settingsWidget {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """
        )

    def __setValues(self):
        """设置初始值"""
        # 设置字幕排布
        self.layoutCard.comboBox.setCurrentText(
            LAYOUT_VALUE_TO_LABEL.get(cfg.get(cfg.subtitle_layout), "Перевод сверху")
        )
        # 设置字幕效果
        current_effect = cfg.get(cfg.subtitle_effect)
        current_effect_label = next(
            (
                label
                for label, value in self.effect_options.items()
                if value == current_effect
            ),
            "Без эффекта",
        )
        self.effectCard.comboBox.setCurrentText(current_effect_label)
        self.effectDurationCard.spinBox.setValue(cfg.get(cfg.subtitle_effect_duration))
        self.effectIntensityCard.spinBox.setValue(cfg.get(cfg.subtitle_effect_intensity))
        rainbow_hex = cfg.get(cfg.subtitle_rainbow_end_color) or "#0000FF"
        self.rainbowEndColorCard.setColor(QColor(rainbow_hex))
        self.presetCard.comboBox.setCurrentText(
            STYLE_PRESET_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_style_preset), "Свой (Custom)"
            )
        )
        self.motionDirectionCard.comboBox.setCurrentText(
            MOTION_DIRECTION_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_motion_direction), "Снизу вверх"
            )
        )
        self.motionAmplitudeCard.spinBox.setValue(
            int(cfg.get(cfg.subtitle_motion_amplitude))
        )
        self.motionEasingCard.comboBox.setCurrentText(
            MOTION_EASING_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_motion_easing), "Ease Out (мягкий финиш)"
            )
        )
        self.motionJitterCard.spinBox.setValue(int(cfg.get(cfg.subtitle_motion_jitter)))
        self.karaokeModeCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_karaoke_mode)), "Выкл")
        )
        self.karaokeWindowCard.spinBox.setValue(
            int(cfg.get(cfg.subtitle_karaoke_window_ms))
        )
        self.needSplitCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.need_split)), "Выкл")
        )
        self.splitTypeCard.comboBox.setCurrentText(
            SPLIT_TYPE_VALUE_TO_LABEL.get(
                cfg.get(cfg.split_type),
                "По предложениям",
            )
        )
        self._refresh_effect_options_by_split_mode()
        self._refresh_preview_text_options()
        self.maxWordCountCjkCard.spinBox.setValue(int(cfg.get(cfg.max_word_count_cjk)))
        self.maxWordCountEnglishCard.spinBox.setValue(
            int(cfg.get(cfg.max_word_count_english))
        )
        self.removePunctuationCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(
                bool(cfg.get(cfg.needs_remove_punctuation)),
                "Вкл",
            )
        )
        self.autoContrastCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_auto_contrast)), "Выкл")
        )
        self.antiFlickerCard.comboBox.setCurrentText(
            TOGGLE_VALUE_TO_LABEL.get(bool(cfg.get(cfg.subtitle_anti_flicker)), "Вкл")
        )
        self.gradientModeCard.comboBox.setCurrentText(
            GRADIENT_MODE_VALUE_TO_LABEL.get(
                cfg.get(cfg.subtitle_gradient_mode), "Без градиента"
            )
        )
        self.gradientColor1Card.setColor(
            QColor(cfg.get(cfg.subtitle_gradient_color_1) or "#FFFFFF")
        )
        self.gradientColor2Card.setColor(
            QColor(cfg.get(cfg.subtitle_gradient_color_2) or "#66CCFF")
        )
        # 设置字幕样式
        self.styleNameComboBox.comboBox.setCurrentText(cfg.get(cfg.subtitle_style_name))

        # 获取系统字体,设置comboBox的选项
        fontDatabase = QFontDatabase()
        fontFamilies = fontDatabase.families()
        self.mainFontCard.addItems(fontFamilies)
        self.subFontCard.addItems(fontFamilies)

        # 设置字体选项框最大显示数量
        self.mainFontCard.comboBox.setMaxVisibleItems(12)
        self.subFontCard.comboBox.setMaxVisibleItems(12)

        # 获取样式目录下的所有txt文件名
        style_files = [f.stem for f in SUBTITLE_STYLE_PATH.glob("*.txt")]
        if "default" in style_files:
            style_files.insert(0, style_files.pop(style_files.index("default")))
        else:
            style_files.insert(0, "default")
            self.saveStyle("default")
        self.styleNameComboBox.comboBox.addItems(style_files)

        # 加载默认样式
        subtitle_style_name = cfg.get(cfg.subtitle_style_name)
        if subtitle_style_name in style_files:
            self.loadStyle(subtitle_style_name)
            self.styleNameComboBox.comboBox.setCurrentText(subtitle_style_name)
        else:
            self.loadStyle(style_files[0])
            self.styleNameComboBox.comboBox.setCurrentText(style_files[0])

    def connectSignals(self):
        """连接所有设置变更的信号到预览更新函数"""
        # 字幕排布
        self.layoutCard.currentTextChanged.connect(self.onSettingChanged)
        self.layoutCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_layout,
                LAYOUT_LABEL_TO_VALUE.get(text, "译文在上"),
            )
        )
        self.effectCard.currentTextChanged.connect(self.onSettingChanged)
        self.effectCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_effect,
                self.effect_options.get(text, "none"),
            )
        )
        self.effectCard.currentTextChanged.connect(
            lambda _: self._update_motion_controls_state()
        )
        self.effectDurationCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.effectDurationCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_effect_duration, int(value))
        )
        self.effectIntensityCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.effectIntensityCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_effect_intensity, int(value))
        )
        self.rainbowEndColorCard.colorChanged.connect(self.onSettingChanged)
        self.rainbowEndColorCard.colorChanged.connect(
            lambda color: cfg.set(cfg.subtitle_rainbow_end_color, color.name())
        )
        self.presetCard.currentTextChanged.connect(self.onPresetChanged)
        self.presetCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_style_preset,
                STYLE_PRESET_LABEL_TO_VALUE.get(text, "custom"),
            )
        )
        self.motionDirectionCard.currentTextChanged.connect(self.onSettingChanged)
        self.motionDirectionCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_motion_direction,
                MOTION_DIRECTION_LABEL_TO_VALUE.get(text, "up"),
            )
        )
        self.motionAmplitudeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.motionAmplitudeCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_motion_amplitude, int(value))
        )
        self.motionEasingCard.currentTextChanged.connect(self.onSettingChanged)
        self.motionEasingCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_motion_easing,
                MOTION_EASING_LABEL_TO_VALUE.get(text, "ease_out"),
            )
        )
        self.motionJitterCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.motionJitterCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_motion_jitter, int(value))
        )
        self.karaokeModeCard.currentTextChanged.connect(self.onSettingChanged)
        self.karaokeModeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_karaoke_mode,
                TOGGLE_LABEL_TO_VALUE.get(text, False),
            )
        )
        self.karaokeWindowCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.karaokeWindowCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.subtitle_karaoke_window_ms, int(value))
        )
        self.needSplitCard.currentTextChanged.connect(self.onSettingChanged)
        self.needSplitCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.need_split,
                TOGGLE_LABEL_TO_VALUE.get(text, False),
            )
        )
        self.splitTypeCard.currentTextChanged.connect(self._on_split_type_changed)
        self.splitTypeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.split_type,
                SPLIT_TYPE_LABEL_TO_VALUE.get(text, SplitTypeEnum.SENTENCE.value),
            )
        )
        self.maxWordCountCjkCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.maxWordCountCjkCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.max_word_count_cjk, int(value))
        )
        self.maxWordCountEnglishCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.maxWordCountEnglishCard.spinBox.valueChanged.connect(
            lambda value: cfg.set(cfg.max_word_count_english, int(value))
        )
        self.removePunctuationCard.currentTextChanged.connect(self.onSettingChanged)
        self.removePunctuationCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.needs_remove_punctuation,
                TOGGLE_LABEL_TO_VALUE.get(text, True),
            )
        )
        self.autoContrastCard.currentTextChanged.connect(self.onSettingChanged)
        self.autoContrastCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_auto_contrast,
                TOGGLE_LABEL_TO_VALUE.get(text, False),
            )
        )
        self.antiFlickerCard.currentTextChanged.connect(self.onSettingChanged)
        self.antiFlickerCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_anti_flicker,
                TOGGLE_LABEL_TO_VALUE.get(text, True),
            )
        )
        self.gradientModeCard.currentTextChanged.connect(self.onSettingChanged)
        self.gradientModeCard.currentTextChanged.connect(
            lambda text: cfg.set(
                cfg.subtitle_gradient_mode,
                GRADIENT_MODE_LABEL_TO_VALUE.get(text, "off"),
            )
        )
        self.gradientColor1Card.colorChanged.connect(self.onSettingChanged)
        self.gradientColor1Card.colorChanged.connect(
            lambda color: cfg.set(cfg.subtitle_gradient_color_1, color.name())
        )
        self.gradientColor2Card.colorChanged.connect(self.onSettingChanged)
        self.gradientColor2Card.colorChanged.connect(
            lambda color: cfg.set(cfg.subtitle_gradient_color_2, color.name())
        )
        # 垂直间距
        self.verticalSpacingCard.spinBox.valueChanged.connect(self.onSettingChanged)

        # 主字幕样式
        self.mainFontCard.currentTextChanged.connect(self.onSettingChanged)
        self.mainSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainSpacingCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainColorCard.colorChanged.connect(self.onSettingChanged)
        self.mainOutlineColorCard.colorChanged.connect(self.onSettingChanged)
        self.mainOutlineSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainShadowCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.mainShadowColorCard.colorChanged.connect(self.onSettingChanged)
        self.mainBlurCard.spinBox.valueChanged.connect(self.onSettingChanged)

        # 副字幕样式
        self.subFontCard.currentTextChanged.connect(self.onSettingChanged)
        self.subSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subSpacingCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subColorCard.colorChanged.connect(self.onSettingChanged)
        self.subOutlineColorCard.colorChanged.connect(self.onSettingChanged)
        self.subOutlineSizeCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subShadowCard.spinBox.valueChanged.connect(self.onSettingChanged)
        self.subShadowColorCard.colorChanged.connect(self.onSettingChanged)
        self.subBlurCard.spinBox.valueChanged.connect(self.onSettingChanged)

        # 预览设置
        self.previewTextCard.currentTextChanged.connect(self.onSettingChanged)
        self.orientationCard.currentTextChanged.connect(self.onOrientationChanged)
        self.previewImageCard.clicked.connect(self.selectPreviewImage)
        self.previewEffectButton.clicked.connect(self.showEffectPreview)
        self.playPreviewButton.clicked.connect(self.toggleLivePreview)
        self.timelineSlider.valueChanged.connect(self.onTimelineChanged)

        # 连接样式切换信号
        self.styleNameComboBox.currentTextChanged.connect(self.loadStyle)
        self.newStyleButton.clicked.connect(self.createNewStyle)
        self.openStyleFolderButton.clicked.connect(self.on_open_style_folder_clicked)

        # 连接字幕排布信号
        self.layoutCard.comboBox.currentTextChanged.connect(
            lambda text: signalBus.subtitle_layout_changed.emit(
                LAYOUT_LABEL_TO_VALUE.get(text, "译文在上")
            )
        )
        signalBus.subtitle_layout_changed.connect(self.on_subtitle_layout_changed)

    def on_open_style_folder_clicked(self):
        """打开样式文件夹"""
        if sys.platform == "win32":
            os.startfile(SUBTITLE_STYLE_PATH)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", SUBTITLE_STYLE_PATH])
        else:  # Linux
            subprocess.run(["xdg-open", SUBTITLE_STYLE_PATH])

    def on_subtitle_layout_changed(self, layout: str):
        cfg.subtitle_layout.value = layout
        self.layoutCard.setCurrentText(LAYOUT_VALUE_TO_LABEL.get(layout, "Перевод сверху"))

    def onOrientationChanged(self):
        """当预览方向改变时调用"""
        orientation_label = self.orientationCard.comboBox.currentText()
        orientation = ORIENTATION_LABEL_TO_VALUE.get(orientation_label, "横屏")
        preview_image = (
            DEFAULT_BG_LANDSCAPE if orientation == "横屏" else DEFAULT_BG_PORTRAIT
        )
        cfg.set(cfg.subtitle_preview_image, str(Path(preview_image["path"])))
        self._schedulePreviewUpdate()

    def onSettingChanged(self):
        """当任何设置改变时调用"""
        # 如果正在加载样式，不触发更新
        if self._loading_style:
            return

        self._update_word_timestamp_hint()

        self._schedulePreviewUpdate()
        # 获取当前选择的样式名称
        current_style = self.styleNameComboBox.comboBox.currentText()
        if current_style:
            self.saveStyle(current_style)  # 自动保存为当前选择的样式
        else:
            self.saveStyle("default")  # 如果没有选择样式,保存为默认样式

    def _on_split_type_changed(self):
        self._refresh_effect_options_by_split_mode()
        self._refresh_preview_text_options()
        self._update_word_timestamp_hint()
        self.onSettingChanged()

    def _is_word_mode(self) -> bool:
        return self.splitTypeCard.comboBox.currentText() == "По словам"

    def _get_allowed_effect_values(self) -> set[str]:
        return WORD_MODE_EFFECTS if self._is_word_mode() else SENTENCE_MODE_EFFECTS

    def _refresh_effect_options_by_split_mode(self):
        allowed = self._get_allowed_effect_values()
        current_label = self.effectCard.comboBox.currentText()

        filtered = [
            label
            for label, value in self.effect_options.items()
            if value in allowed
        ]
        if not filtered:
            filtered = ["Без эффекта"]

        self.effectCard.comboBox.blockSignals(True)
        self.effectCard.comboBox.clear()
        self.effectCard.comboBox.addItems(filtered)
        if current_label in filtered:
            self.effectCard.comboBox.setCurrentText(current_label)
        else:
            self.effectCard.comboBox.setCurrentIndex(0)
        self.effectCard.comboBox.blockSignals(False)

        # Применяем значение в cfg после фильтрации
        selected_label = self.effectCard.comboBox.currentText()
        cfg.set(cfg.subtitle_effect, self.effect_options.get(selected_label, "none"))
        self._update_motion_controls_state()

    def _current_preview_texts(self):
        split_label = self.splitTypeCard.comboBox.currentText()
        if split_label == "По словам":
            return PREVIEW_TEXTS_WORD
        return PREVIEW_TEXTS_SENTENCE

    def _refresh_preview_text_options(self):
        texts_map = self._current_preview_texts()
        current = self.previewTextCard.comboBox.currentText()
        self.previewTextCard.comboBox.blockSignals(True)
        self.previewTextCard.comboBox.clear()
        self.previewTextCard.comboBox.addItems(list(texts_map.keys()))
        if current in texts_map:
            self.previewTextCard.comboBox.setCurrentText(current)
        else:
            self.previewTextCard.comboBox.setCurrentIndex(0)
        self.previewTextCard.comboBox.blockSignals(False)

    def _update_motion_controls_state(self):
        effect_label = self.effectCard.comboBox.currentText()
        effect_value = self.effect_options.get(effect_label, "none")
        is_word_mode = self._is_word_mode()
        enabled = is_word_mode and EffectManager.is_motion_customizable(effect_value)
        self.karaokeModeCard.setEnabled(not is_word_mode)
        self.karaokeWindowCard.setEnabled(not is_word_mode)

        self.motionDirectionCard.setEnabled(enabled)
        self.motionAmplitudeCard.setEnabled(enabled)
        self.motionEasingCard.setEnabled(enabled)
        self.motionJitterCard.setEnabled(enabled)
        self._update_word_timestamp_hint()

    def _update_word_timestamp_hint(self):
        split_is_word = self.splitTypeCard.comboBox.currentText() == "По словам"
        effect_value = self.effect_options.get(self.effectCard.comboBox.currentText(), "none")
        effect_is_word = effect_value == "word_highlight"
        karaoke_on = TOGGLE_LABEL_TO_VALUE.get(
            self.karaokeModeCard.comboBox.currentText(), False
        )
        need_hint = split_is_word or effect_is_word or karaoke_on
        self.wordTimestampHintLabel.setVisible(need_hint)

    def toggleLivePreview(self):
        self._live_playing = not self._live_playing
        if self._live_playing:
            self.playPreviewButton.setText("⏸ Live")
            self._live_timer.start()
        else:
            self.playPreviewButton.setText("▶ Live")
            self._live_timer.stop()

    def _onLiveTimerTick(self):
        next_value = self.timelineSlider.value() + self.timelineSlider.singleStep()
        if next_value > self.timelineSlider.maximum():
            next_value = self.timelineSlider.minimum()
        self.timelineSlider.setValue(next_value)

    def onTimelineChanged(self, value: int):
        self.timelineLabel.setText(f"{value / 1000:.2f} c")
        self._schedulePreviewUpdate(immediate=self._live_playing)

    def onPresetChanged(self, preset_label: str):
        if self._loading_style:
            return

        preset = STYLE_PRESET_LABEL_TO_VALUE.get(preset_label, "custom")
        if preset == "custom":
            self.onSettingChanged()
            return

        self._loading_style = True
        try:
            if preset == "tiktok_dynamic":
                self.effectCard.comboBox.setCurrentText("Подсветка по словам")
                self.effectDurationCard.spinBox.setValue(450)
                self.effectIntensityCard.spinBox.setValue(140)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(140)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(12)
                self.mainOutlineSizeCard.spinBox.setValue(2.8)
                self.mainShadowCard.spinBox.setValue(2.2)
            elif preset == "shorts_clean":
                self.effectCard.comboBox.setCurrentText("Плавное появление")
                self.effectDurationCard.spinBox.setValue(260)
                self.effectIntensityCard.spinBox.setValue(90)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(80)
                self.motionEasingCard.comboBox.setCurrentText("Linear")
                self.motionJitterCard.spinBox.setValue(0)
                self.mainOutlineSizeCard.spinBox.setValue(2.0)
                self.mainShadowCard.spinBox.setValue(1.2)
            elif preset == "minimal_classic":
                self.effectCard.comboBox.setCurrentText("Без эффекта")
                self.effectDurationCard.spinBox.setValue(200)
                self.effectIntensityCard.spinBox.setValue(100)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(60)
                self.motionEasingCard.comboBox.setCurrentText("Ease In-Out")
                self.motionJitterCard.spinBox.setValue(0)
                self.mainOutlineSizeCard.spinBox.setValue(1.6)
                self.mainShadowCard.spinBox.setValue(0.8)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Выкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("Без градиента")
            elif preset == "karaoke_pro":
                self.effectCard.comboBox.setCurrentText("Подсветка по словам")
                self.effectDurationCard.spinBox.setValue(600)
                self.effectIntensityCard.spinBox.setValue(130)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(110)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(6)
                self.karaokeModeCard.comboBox.setCurrentText("Вкл")
                self.karaokeWindowCard.spinBox.setValue(1400)
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#FFF4B2"))
                self.gradientColor2Card.setColor(QColor("#FF6FAE"))
            elif preset == "cinema_gradient":
                self.effectCard.comboBox.setCurrentText("Плавное появление")
                self.effectDurationCard.spinBox.setValue(380)
                self.effectIntensityCard.spinBox.setValue(105)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(85)
                self.motionEasingCard.comboBox.setCurrentText("Ease In-Out")
                self.motionJitterCard.spinBox.setValue(0)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("2 цвета")
                self.gradientColor1Card.setColor(QColor("#E8F3FF"))
                self.gradientColor2Card.setColor(QColor("#6AB8FF"))
            elif preset == "neon_pulse":
                self.effectCard.comboBox.setCurrentText("Неоновое мерцание")
                self.effectDurationCard.spinBox.setValue(520)
                self.effectIntensityCard.spinBox.setValue(150)
                self.motionDirectionCard.comboBox.setCurrentText("Снизу вверх")
                self.motionAmplitudeCard.spinBox.setValue(130)
                self.motionEasingCard.comboBox.setCurrentText("Ease Out (мягкий финиш)")
                self.motionJitterCard.spinBox.setValue(10)
                self.karaokeModeCard.comboBox.setCurrentText("Выкл")
                self.autoContrastCard.comboBox.setCurrentText("Вкл")
                self.antiFlickerCard.comboBox.setCurrentText("Вкл")
                self.gradientModeCard.comboBox.setCurrentText("Радужный")
        finally:
            self._loading_style = False

        self._schedulePreviewUpdate()
        current_style = self.styleNameComboBox.comboBox.currentText() or "default"
        self.saveStyle(current_style)

    def selectPreviewImage(self):
        """选择预览背景图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите фоновое изображение",
            "",
            "Файлы изображений (*.png *.jpg *.jpeg)",
        )
        if file_path:
            cfg.set(cfg.subtitle_preview_image, file_path)
            self._schedulePreviewUpdate()

    def _schedulePreviewUpdate(self, immediate: bool = False):
        self._preview_pending = True
        if immediate or self._live_playing:
            if self._preview_thread and self._preview_thread.isRunning():
                return
            self._renderPreviewNow()
            return
        self._preview_debounce_timer.start()

    def _renderPreviewNow(self):
        if not self._preview_pending:
            return

        if self._preview_thread and self._preview_thread.isRunning():
            return

        self._preview_pending = False
        self._preview_request_id += 1
        request_id = self._preview_request_id

        # 生成 ASS 样式字符串
        style_str = self.generateAssStyles()

        # 获取预览文本
        preview_texts = self._current_preview_texts()
        main_text, sub_text = preview_texts[self.previewTextCard.comboBox.currentText()]

        # 字幕布局
        layout = self.layoutCard.comboBox.currentText()
        layout_value = LAYOUT_LABEL_TO_VALUE.get(layout, "译文在上")
        if layout_value == "译文在上":
            main_text, sub_text = sub_text, main_text
        elif layout_value == "原文在上":
            main_text, sub_text = main_text, sub_text
        elif layout_value == "仅译文":
            main_text, sub_text = sub_text, None
        elif layout_value == "仅原文":
            main_text, sub_text = main_text, None

        # 获取预览方向
        orientation_label = self.orientationCard.comboBox.currentText()
        orientation = ORIENTATION_LABEL_TO_VALUE.get(orientation_label, "横屏")
        default_preview = (
            DEFAULT_BG_LANDSCAPE if orientation == "横屏" else DEFAULT_BG_PORTRAIT
        )

        # 检查是否存在用户自定义背景图片
        user_bg_path = cfg.get(cfg.subtitle_preview_image)
        if user_bg_path and Path(user_bg_path).exists():
            path = user_bg_path
            width = default_preview["width"]
            height = default_preview["height"]
        else:
            path = default_preview["path"]
            width = default_preview["width"]
            height = default_preview["height"]

        timeline_ms = self.timelineSlider.value()
        effect_type = cfg.get(cfg.subtitle_effect)
        if effect_type != "none":
            preview_window_ms = self.timelineSlider.maximum()
            effect_ms = min(
                max(50, int(cfg.get(cfg.subtitle_effect_duration))),
                preview_window_ms,
            )
            preview_time_sec = (
                (timeline_ms / max(1, self.timelineSlider.maximum()))
                * (effect_ms / 1000.0)
            )
        else:
            preview_time_sec = timeline_ms / 1000.0

        self._preview_thread = PreviewThread(
            style_str=style_str,
            preview_text=(main_text, sub_text),
            bg_path=path,
            width=width,
            height=height,
            effect_type=effect_type,
            effect_duration_ms=cfg.get(cfg.subtitle_effect_duration),
            effect_intensity=cfg.get(cfg.subtitle_effect_intensity) / 100,
            rainbow_end_color=cfg.get(cfg.subtitle_rainbow_end_color),
            motion_direction=cfg.get(cfg.subtitle_motion_direction),
            motion_amplitude=cfg.get(cfg.subtitle_motion_amplitude) / 100,
            motion_easing=cfg.get(cfg.subtitle_motion_easing),
            motion_jitter=cfg.get(cfg.subtitle_motion_jitter) / 100,
            karaoke_mode=cfg.get(cfg.subtitle_karaoke_mode),
            karaoke_window_ms=cfg.get(cfg.subtitle_karaoke_window_ms),
            auto_contrast=cfg.get(cfg.subtitle_auto_contrast),
            anti_flicker=cfg.get(cfg.subtitle_anti_flicker),
            gradient_mode=cfg.get(cfg.subtitle_gradient_mode),
            gradient_color_1=cfg.get(cfg.subtitle_gradient_color_1),
            gradient_color_2=cfg.get(cfg.subtitle_gradient_color_2),
            preview_time_sec=preview_time_sec,
            request_id=request_id,
        )
        self._preview_thread.previewReady.connect(self.onPreviewReady)
        self._preview_thread.finished.connect(self._onPreviewThreadFinished)
        self._preview_thread.start()

    def _onPreviewThreadFinished(self):
        if self._preview_pending:
            self._renderPreviewNow()

    def generateAssStyles(self) -> str:
        """生成 ASS 样式字符串"""
        style_format = "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"

        def _qcolor_to_ass(color: QColor, with_alpha: bool = True) -> str:
            r, g, b, a = color.getRgb()
            # Qt хранит alpha как непрозрачность (0=прозрачный, 255=непрозрачный),
            # а ASS хранит alpha как прозрачность (00=непрозрачный, FF=прозрачный)
            ass_alpha = 255 - a if with_alpha else 0
            aa = f"{ass_alpha:02X}"
            return f"&H{aa}{b:02X}{g:02X}{r:02X}"

        # 从控件获取当前设置
        # 获取垂直间距
        vertical_spacing = int(
            self.verticalSpacingCard.spinBox.value()
        )  # 转换为ASS单位

        # 提取主字幕样式元素
        main_font = self.mainFontCard.comboBox.currentText()
        main_size = self.mainSizeCard.spinBox.value()

        # 获取颜色值并转换为 ASS 格式 (AABBGGRR)
        main_color_hex = self.mainColorCard.colorPicker.color.name()
        main_outline_hex = self.mainOutlineColorCard.colorPicker.color.name()
        main_color = (
            f"&H00{main_color_hex[5:7]}{main_color_hex[3:5]}{main_color_hex[1:3]}"
        )
        main_outline_color = (
            f"&H00{main_outline_hex[5:7]}{main_outline_hex[3:5]}{main_outline_hex[1:3]}"
        )
        main_spacing = self.mainSpacingCard.spinBox.value()
        main_outline_size = self.mainOutlineSizeCard.spinBox.value()
        main_shadow = self.mainShadowCard.spinBox.value()
        main_shadow_color = _qcolor_to_ass(self.mainShadowColorCard.colorPicker.color)
        main_blur = self.mainBlurCard.spinBox.value()

        # 提取副字幕样式元素
        sub_font = self.subFontCard.comboBox.currentText()
        sub_size = self.subSizeCard.spinBox.value()

        # 获取颜色值并转换为 ASS 格式 (AABBGGRR)
        sub_color_hex = self.subColorCard.colorPicker.color.name()
        sub_outline_hex = self.subOutlineColorCard.colorPicker.color.name()
        sub_color = f"&H00{sub_color_hex[5:7]}{sub_color_hex[3:5]}{sub_color_hex[1:3]}"
        sub_outline_color = (
            f"&H00{sub_outline_hex[5:7]}{sub_outline_hex[3:5]}{sub_outline_hex[1:3]}"
        )
        sub_spacing = self.subSpacingCard.spinBox.value()
        sub_outline_size = self.subOutlineSizeCard.spinBox.value()
        sub_shadow = self.subShadowCard.spinBox.value()
        sub_shadow_color = _qcolor_to_ass(self.subShadowColorCard.colorPicker.color)
        sub_blur = self.subBlurCard.spinBox.value()

        # 生成样式字符串
        # В ASS нет отдельного поля blur в строке Style, поэтому храним его
        # как комментарии вида ;VC_BLUR:StyleName=value и применяем в Dialogue.
        main_style = f"Style: Default,{main_font},{main_size},{main_color},&H000000FF,{main_outline_color},{main_shadow_color},-1,0,0,0,100,100,{main_spacing},0,1,{main_outline_size},{main_shadow},2,10,10,{vertical_spacing},1"
        sub_style = f"Style: Secondary,{sub_font},{sub_size},{sub_color},&H000000FF,{sub_outline_color},{sub_shadow_color},-1,0,0,0,100,100,{sub_spacing},0,1,{sub_outline_size},{sub_shadow},2,10,10,{vertical_spacing},1"

        extras = [
            f";VC_BLUR:Default={main_blur}",
            f";VC_BLUR:Secondary={sub_blur}",
        ]

        return (
            f"[V4+ Styles]\n{style_format}\n{main_style}\n{sub_style}\n"
            + "\n".join(extras)
        )

    def updatePreview(self):
        """请求更新预览（带防抖和串行渲染）"""
        self._schedulePreviewUpdate()

    def showEffectPreview(self):
        """Сгенерировать короткое видео предпросмотра эффекта и открыть его."""
        style_str = self.generateAssStyles()
        preview_texts = self._current_preview_texts()
        main_text, sub_text = preview_texts[self.previewTextCard.comboBox.currentText()]

        layout_value = LAYOUT_LABEL_TO_VALUE.get(
            self.layoutCard.comboBox.currentText(), "译文在上"
        )
        if layout_value == "译文在上":
            main_text, sub_text = sub_text, main_text
        elif layout_value == "原文在上":
            pass
        elif layout_value == "仅译文":
            main_text, sub_text = sub_text, None
        elif layout_value == "仅原文":
            sub_text = None

        orientation = ORIENTATION_LABEL_TO_VALUE.get(
            self.orientationCard.comboBox.currentText(), "横屏"
        )
        default_preview = DEFAULT_BG_LANDSCAPE if orientation == "横屏" else DEFAULT_BG_PORTRAIT
        user_bg_path = cfg.get(cfg.subtitle_preview_image)
        path = user_bg_path if user_bg_path and Path(user_bg_path).exists() else default_preview["path"]

        video_path = generate_preview_video(
            style_str=style_str,
            preview_text=(main_text, sub_text),
            bg_path=str(path),
            width=default_preview["width"],
            height=default_preview["height"],
            effect_type=cfg.get(cfg.subtitle_effect),
            effect_duration_ms=cfg.get(cfg.subtitle_effect_duration),
            effect_intensity=cfg.get(cfg.subtitle_effect_intensity) / 100,
            rainbow_end_color=cfg.get(cfg.subtitle_rainbow_end_color),
            motion_direction=cfg.get(cfg.subtitle_motion_direction),
            motion_amplitude=cfg.get(cfg.subtitle_motion_amplitude) / 100,
            motion_easing=cfg.get(cfg.subtitle_motion_easing),
            motion_jitter=cfg.get(cfg.subtitle_motion_jitter) / 100,
            karaoke_mode=cfg.get(cfg.subtitle_karaoke_mode),
            karaoke_window_ms=cfg.get(cfg.subtitle_karaoke_window_ms),
            auto_contrast=cfg.get(cfg.subtitle_auto_contrast),
            anti_flicker=cfg.get(cfg.subtitle_anti_flicker),
            gradient_mode=cfg.get(cfg.subtitle_gradient_mode),
            gradient_color_1=cfg.get(cfg.subtitle_gradient_color_1),
            gradient_color_2=cfg.get(cfg.subtitle_gradient_color_2),
        )

        if sys.platform == "win32":
            os.startfile(video_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", video_path])
        else:
            subprocess.run(["xdg-open", video_path])

    def onPreviewReady(self, preview_path, request_id: int):
        """预览图片生成完成的回调"""
        if request_id < self._latest_applied_request_id:
            return
        self._latest_applied_request_id = request_id
        self.previewImage.setImage(preview_path)
        self.updatePreviewImage()

    def updatePreviewImage(self):
        """更新预览图片"""
        height = int(self.previewTopWidget.height() * 0.98)
        width = int(self.previewTopWidget.width() * 0.98)
        self.previewImage.scaledToWidth(width)
        if self.previewImage.height() > height:
            self.previewImage.scaledToHeight(height)
        self.previewImage.setBorderRadius(8, 8, 8, 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updatePreviewImage()

    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        self.updatePreviewImage()

    def closeEvent(self, event):
        self._live_timer.stop()
        self._preview_debounce_timer.stop()
        if self._preview_thread and self._preview_thread.isRunning():
            self._preview_thread.quit()
            if not self._preview_thread.wait(500):
                self._preview_thread.terminate()
                self._preview_thread.wait(200)
        super().closeEvent(event)

    def loadStyle(self, style_name):
        """加载指定样式"""
        style_path = SUBTITLE_STYLE_PATH / f"{style_name}.txt"

        if not style_path.exists():
            return

        # 设置标志位，防止触发onSettingChanged
        self._loading_style = True

        with open(style_path, "r", encoding="utf-8") as f:
            style_content = f.read()

        # 解析样式内容
        for line in style_content.split("\n"):
            if line.startswith("Style: Default"):
                # 解析主字幕样式
                parts = line.split(",")
                self.mainFontCard.setCurrentText(parts[1])
                self.mainSizeCard.spinBox.setValue(int(parts[2]))

                vertical_spacing = int(parts[21])
                self.verticalSpacingCard.spinBox.setValue(vertical_spacing)

                # 将 &HAARRGGBB 格式转换为 QColor
                primary_color = parts[3].strip()
                if primary_color.startswith("&H"):
                    # 移除 &H 前缀,转换为 RGB
                    color_hex = primary_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.mainColorCard.setColor(QColor(red, green, blue, alpha))

                outline_color = parts[5].strip()
                if outline_color.startswith("&H"):
                    color_hex = outline_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.mainOutlineColorCard.setColor(QColor(red, green, blue, alpha))

                shadow_color = parts[6].strip()
                if shadow_color.startswith("&H") and len(shadow_color) >= 10:
                    color_hex = shadow_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.mainShadowColorCard.setColor(QColor(red, green, blue, alpha))

                self.mainSpacingCard.spinBox.setValue(float(parts[13]))
                self.mainOutlineSizeCard.spinBox.setValue(float(parts[16]))
                self.mainShadowCard.spinBox.setValue(float(parts[17]))
                blur_match = re.search(r";VC_BLUR:Default=([0-9.]+)", style_content)
                if not blur_match:
                    blur_match = re.search(r"#EXTRAS:blur=([0-9.]+)", style_content)
                self.mainBlurCard.spinBox.setValue(
                    float(blur_match.group(1)) if blur_match else 0.0
                )
            elif line.startswith("Style: Secondary"):
                # 解析副字幕样式
                parts = line.split(",")
                self.subFontCard.setCurrentText(parts[1])
                self.subSizeCard.spinBox.setValue(int(parts[2]))
                # 将 &HAARRGGBB 格式转换为 QColor
                primary_color = parts[3].strip()
                if primary_color.startswith("&H"):
                    color_hex = primary_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.subColorCard.setColor(QColor(red, green, blue, alpha))

                outline_color = parts[5].strip()
                if outline_color.startswith("&H"):
                    color_hex = outline_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.subOutlineColorCard.setColor(QColor(red, green, blue, alpha))

                shadow_color = parts[6].strip()
                if shadow_color.startswith("&H") and len(shadow_color) >= 10:
                    color_hex = shadow_color[2:]
                    alpha = 255 - int(color_hex[0:2], 16)
                    blue = int(color_hex[2:4], 16)
                    green = int(color_hex[4:6], 16)
                    red = int(color_hex[6:8], 16)
                    self.subShadowColorCard.setColor(QColor(red, green, blue, alpha))

                self.subSpacingCard.spinBox.setValue(float(parts[13]))
                self.subOutlineSizeCard.spinBox.setValue(float(parts[16]))
                self.subShadowCard.spinBox.setValue(float(parts[17]))
                blur_match = re.search(r";VC_BLUR:Secondary=([0-9.]+)", style_content)
                if not blur_match:
                    blur_match = re.search(r"#EXTRAS:blur=([0-9.]+)", style_content)
                self.subBlurCard.spinBox.setValue(
                    float(blur_match.group(1)) if blur_match else 0.0
                )

        cfg.set(cfg.subtitle_style_name, style_name)

        # 重置标志位
        self._loading_style = False

        # 手动更新一次预览
        self.updatePreview()

        # 显示加载成功提示
        InfoBar.success(
            title="Успешно",
            content="Загружен стиль: " + style_name,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=1500,
            parent=self,
        )

    def createNewStyle(self):
        """创建新样式"""
        dialog = StyleNameDialog(self)
        if dialog.exec():
            style_name = dialog.nameLineEdit.text().strip()
            if not style_name:
                return

            # 检查是否已存在同名样式
            if (SUBTITLE_STYLE_PATH / f"{style_name}.txt").exists():
                InfoBar.warning(
                    title="Внимание",
                    content="Стиль уже существует: " + style_name,
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self,
                )
                return

            # 保存新样式
            self.saveStyle(style_name)

            # 更新样式列表并选中新样式
            self.styleNameComboBox.addItem(style_name)
            self.styleNameComboBox.comboBox.setCurrentText(style_name)

            # 显示创建成功提示
            InfoBar.success(
                title="Успешно",
                content="Создан новый стиль: " + style_name,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )

    def saveStyle(self, style_name):
        """保存样式
        Args:
            style_name (str): 样式名称
        """
        # 确保样式目录存在
        SUBTITLE_STYLE_PATH.mkdir(parents=True, exist_ok=True)

        # 生成样式内容并保存
        style_content = self.generateAssStyles()
        style_path = SUBTITLE_STYLE_PATH / f"{style_name}.txt"

        with open(style_path, "w", encoding="utf-8") as f:
            f.write(style_content)


class StyleNameDialog(MessageBoxBase):
    """样式名称输入对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel("Новый стиль", self)
        self.nameLineEdit = LineEdit(self)

        self.nameLineEdit.setPlaceholderText("Введите название стиля")
        self.nameLineEdit.setClearButtonEnabled(True)

        # 添加控件到布局
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameLineEdit)

        # 设置按钮文本
        self.yesButton.setText("ОК")
        self.cancelButton.setText("Отмена")

        self.widget.setMinimumWidth(350)
        self.yesButton.setDisabled(True)
        self.nameLineEdit.textChanged.connect(self._validateInput)

    def _validateInput(self, text):
        self.yesButton.setEnabled(bool(text.strip()))
