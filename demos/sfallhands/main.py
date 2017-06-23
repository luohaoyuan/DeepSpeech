#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import csv
import os
import sys
import textwrap

from threading import Thread
from queue import Queue

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))

import tensorflow as tf

from util.audio import audiofile_to_input_vector
from util.text import ndarray_to_text
from util.spell import correction

n_input = 26
n_context = 9

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import *

class Corpus(object):
    def __init__(self, name, checkpoint_path):
        self._name = name
        self._checkpoint_path = checkpoint_path

    @property
    def name(self):
        return self._name

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

class Sample(QObject):
    def __init__(self, wav_path, transcription, source, extra_text, color):
        super().__init__()
        self._wav_path = wav_path
        self._transcription = transcription
        self._source = source
        self._extra_text = extra_text
        self._color = color
        self._button = None

    @property
    def wav_path(self):
        return self._wav_path

    @property
    def transcription(self):
        return self._transcription

    @property
    def source(self):
        return self._source

    @property
    def extra_text(self):
        return self._extra_text

    @property
    def color(self):
        return self._color

    def set_button(self, button):
        self._button = button

    @property
    def button(self):
        return self._button

class InferenceRunner(QObject):
    inference_done = pyqtSignal(Sample, str)

    def __init__(self, checkpoint_path):
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._queue = Queue()
        self._thread = Thread(target=self._worker_thread)
        self._thread.daemon = True
        self._thread.start()

    def _worker_thread(self):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join('demos', 'sfallhands', 'inference-model.meta'))
        saver.restore(sess, self._checkpoint_path)

        while True:
            cmd, *args = self._queue.get()
            if cmd == 'checkpoint':
                saver.restore(sess, *args)
            elif cmd == 'sample':
                sample, use_LM = args
                vec = audiofile_to_input_vector(sample.wav_path, n_input, n_context)
                output_tensor = sess.graph.get_tensor_by_name('output_node:0')
                result = sess.run([output_tensor], feed_dict={'input_node:0': [vec], 'input_lengths:0': [len(vec)]})
                text = ndarray_to_text(result[0][0][0])
                if use_LM:
                    text = correction(text)
                self.inference_done.emit(sample, text)
            elif cmd == 'stop':
                break

        sess.close()

    def load_checkpoint(self, checkpoint_path):
        self._queue.put(('checkpoint', checkpoint_path))

    def inference(self, sample, use_LM):
        self._queue.put(('sample', sample, use_LM))

    def stop(self):
        self._queue.put(('stop', 'stop'))

class RichTextRadioButton(QRadioButton):
    def __init__(self, richLabel):
        # strip HTML from rich label
        xml = QXmlStreamReader(richLabel)
        plainLabel = ''
        while not xml.atEnd():
            if xml.readNext() == QXmlStreamReader.Characters:
                plainLabel += xml.text()
        super().__init__(plainLabel)
        self._richLabel = richLabel

    def paintEvent(self, event):
        super().paintEvent(event)

        rect = event.rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.eraseRect(rect.topLeft().x()+18, rect.topLeft().y(), rect.width()-18, rect.height())
        painter.translate(QPointF(18, 0))

        label = QTextDocument()
        font = label.defaultFont()
        font.setPixelSize(16)
        label.setDefaultFont(font)
        label.setHtml(self._richLabel)
        label.drawContents(painter)
        painter.end()

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        self._corpora = []
        with open(os.path.join('demos', 'sfallhands', 'corpora.csv'), 'r') as csvfile:
            corpusReader = csv.reader(csvfile)
            next(corpusReader, None) # skip header
            for name, checkpoint_path in corpusReader:
                self._corpora.append(Corpus(name, checkpoint_path))

        self._samples = []
        with open(os.path.join('demos', 'sfallhands', 'samples.csv'), 'r') as csvfile:
            sampleReader = csv.reader(csvfile)
            next(sampleReader, None) # skip header
            for wav_path, transcription, source, extra_text, color in sampleReader:
                self._samples.append(Sample(wav_path, transcription, source, extra_text, color))

        self._inferenceRunner = InferenceRunner(self._corpora[0].checkpoint_path)
        self._inferenceRunner.inference_done.connect(self._on_inference_done)

        self.create_UI()

    def create_UI(self):
        self.resize(1440, 880)
        self.setWindowTitle('Deep Speech Demo')

        quitAction = QAction('Quit')
        quitAction.setShortcut('Ctrl-Q')
        quitAction.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        appMenu = menubar.addMenu('File')
        appMenu.addAction(quitAction)

        corporaButtons = []
        for corpus in self._corpora:
            btn = RichTextRadioButton(corpus.name)
            btn.clicked.connect((lambda c: lambda: self._inferenceRunner.load_checkpoint(c.checkpoint_path))(corpus))
            btn.setStyleSheet('border: 30px; font-size: 20px;')
            corporaButtons.append(btn)

        corporaButtons[0].setChecked(True)

        modelSelectionLabel = QLabel('<p style="font-size:20px; font-style: bold; vertical-align: css bottom;">Use a model trained on the following dataset:</p>')
        modelSelectionLabel.setStyleSheet('max-height: 30px; height: 30px;')
        modelSelectionLabel.setAlignment(Qt.AlignCenter)

        self._useLMButton = QCheckBox("Use Language Model")
        self._useLMButton.setChecked(False)

        modelSelectionHbox = QHBoxLayout()
        modelSelectionHbox.addStretch(1)
        for button in corporaButtons:
            modelSelectionHbox.addWidget(button)
        modelSelectionHbox.addStretch(1)
        modelSelectionHbox.addWidget(self._useLMButton)

        sampleSelectionLabel = QLabel('<p style="font-size:20px; font-style: bold; vertical-align: css bottom;">Click a sample below to hear it and see the transcription from the model:</p>')
        sampleSelectionLabel.setStyleSheet('max-height: 30px; height: 30px;')
        sampleSelectionLabel.setAlignment(Qt.AlignCenter)

        sampleSelectionGrid = QGridLayout()

        positions = [(j, i) for i in range(3) for j in range(5)]
        for i in range(0, min(15, len(self._samples))):
            btn = QPushButton(textwrap.fill(self._samples[i].transcription, 70))
            self._samples[i].set_button(btn)
            btn.clicked.connect((lambda s: lambda: self._sample_clicked(s))(self._samples[i]))
            btn.setStyleSheet('min-height: 100px; min-width: 300px; border: 2px solid ' + self._samples[i].color)
            sampleSelectionGrid.addWidget(btn, *positions[i])

        self._progressBar = QProgressBar(self)
        self._progressBar.setOrientation(Qt.Horizontal)
        self._progressBar.setFormat('Running inference...')
        self._progressBar.setRange(0, 0)
        self._progressBar.setVisible(False)

        self._transcriptionResult = QTextEdit()
        self._transcriptionResult.setReadOnly(True)
        self._transcriptionResult.setStyleSheet('height: 120px;')

        centralWidget = QWidget(self)

        topWidget = QWidget(centralWidget)
        topWidgetLayout = QVBoxLayout()
        topWidgetLayout.addWidget(modelSelectionLabel)
        topWidgetLayout.addLayout(modelSelectionHbox, 3)
        topWidgetLayout.addWidget(sampleSelectionLabel)
        topWidget.setLayout(topWidgetLayout)
        topWidget.setFixedHeight(150)

        bottomWidget = QWidget(centralWidget)
        bottomWidgetLayout = QVBoxLayout()
        bottomWidgetLayout.addWidget(self._progressBar)
        bottomWidgetLayout.addWidget(self._transcriptionResult)
        bottomWidget.setLayout(bottomWidgetLayout)
        bottomWidget.setFixedHeight(130)

        vbox = QVBoxLayout()
        vbox.addWidget(topWidget)
        vbox.addStretch(1)
        vbox.addLayout(sampleSelectionGrid)
        vbox.addStretch(1)
        vbox.addWidget(bottomWidget)

        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)

        self.show()

    def _sample_clicked(self, sample):
        self._progressBar.setVisible(True)
        self._soundEffect = QSoundEffect()
        self._soundEffect.setSource(QUrl.fromLocalFile(sample.wav_path))
        self._soundEffect.setLoopCount(0)
        self._soundEffect.setVolume(1.0)
        self._soundEffect.play()
        sample.button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._soundEffect.playingChanged.connect((lambda sample: lambda: self._on_playing_changed(sample))(sample))
        self._inferenceRunner.inference(sample, self._useLMButton.isChecked())

    def _on_inference_done(self, sample, transcription):
        self._progressBar.setVisible(False)
        self._transcriptionResult.setHtml('<p style="font-size: 20px; text-align: center;">' + transcription + '</p>')

    def _on_playing_changed(self, sample):
        sample.button.setIcon(QIcon())


if __name__ == '__main__':
    app = QApplication(sys.argv)

    demo = MainWidget()

    sys.exit(app.exec_())
