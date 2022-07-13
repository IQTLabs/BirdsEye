#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Birds Eye Squelch
# GNU Radio version: 3.8.3.1
from distutils.version import StrictVersion

if __name__ == "__main__":
    import ctypes
    import sys

    if sys.platform.startswith("linux"):
        try:
            x11 = ctypes.cdll.LoadLibrary("libX11.so")
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
import osmosdr
import time

from gnuradio import qtgui


class birds_eye_squelch_2(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Birds Eye Squelch")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Birds Eye Squelch")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme("gnuradio-grc"))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "birds_eye_squelch_2")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.threshold = threshold = -30
        self.samp_rate = samp_rate = 10500000
        self.gain_0 = gain_0 = 10500000
        self.gain = gain = 45
        self.center_freq = center_freq = 5790000000

        ##################################################
        # Blocks
        ##################################################
        self._threshold_range = Range(-100, 0, 0.5, -30, 200)
        self._threshold_win = RangeWidget(
            self._threshold_range,
            self.set_threshold,
            "threshhold",
            "counter_slider",
            float,
        )
        self.top_layout.addWidget(self._threshold_win)
        self._gain_range = Range(0, 90, 1, 45, 200)
        self._gain_win = RangeWidget(
            self._gain_range, self.set_gain, "gain_0", "counter_slider", int
        )
        self.top_layout.addWidget(self._gain_win)
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024,  # size
            firdes.WIN_BLACKMAN_hARRIS,  # wintype
            center_freq,  # fc
            samp_rate,  # bw
            "",  # name
            1,  # number of inputs
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)

        labels = ["", "", "", "", "", "", "", "", "", ""]
        colors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(
            self.qtgui_waterfall_sink_x_0.pyqwidget(), Qt.QWidget
        )
        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_win)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float, 0.5, qtgui.NUM_GRAPH_HORIZ, 1
        )
        self.qtgui_number_sink_0.set_update_time(0.20)
        self.qtgui_number_sink_0.set_title("")

        labels = ["RSSI", "", "", "", "", "", "", "", "", ""]
        units = ["", "", "", "", "", "", "", "", "", ""]
        colors = [
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
            ("black", "black"),
        ]
        factor = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_0.set_min(i, 0)
            self.qtgui_number_sink_0.set_max(i, 100)
            self.qtgui_number_sink_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels[i])
            self.qtgui_number_sink_0.set_unit(i, units[i])
            self.qtgui_number_sink_0.set_factor(i, factor[i])

        self.qtgui_number_sink_0.enable_autoscale(False)
        self._qtgui_number_sink_0_win = sip.wrapinstance(
            self.qtgui_number_sink_0.pyqwidget(), Qt.QWidget
        )
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024,  # size
            firdes.WIN_HAMMING,  # wintype
            center_freq,  # fc
            samp_rate,  # bw
            "",  # name
            1,
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label("Relative Gain", "dB")
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)

        labels = ["", "", "", "", "", "", "", "", "", ""]
        widths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        colors = [
            "blue",
            "red",
            "green",
            "black",
            "cyan",
            "magenta",
            "yellow",
            "dark red",
            "dark green",
            "dark blue",
        ]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(
            self.qtgui_freq_sink_x_0.pyqwidget(), Qt.QWidget
        )
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.osmosdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + "num_recv_frames=256,recv_frame_size=16360"
        )
        self.osmosdr_source_0.set_sample_rate(samp_rate)
        self.osmosdr_source_0.set_center_freq(center_freq, 0)
        self.osmosdr_source_0.set_freq_corr(0, 0)
        self.osmosdr_source_0.set_dc_offset_mode(0, 0)
        self.osmosdr_source_0.set_iq_balance_mode(0, 0)
        self.osmosdr_source_0.set_gain_mode(False, 0)
        self.osmosdr_source_0.set_gain(gain, 0)
        self.osmosdr_source_0.set_if_gain(0, 0)
        self.osmosdr_source_0.set_bb_gain(0, 0)
        self.osmosdr_source_0.set_antenna("", 0)
        self.osmosdr_source_0.set_bandwidth(samp_rate, 0)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(1024, 10, 40000, 1)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(
            threshold, 5e-4, 1000, True
        )

        ##################################################
        # Connections
        ##################################################
        self.connect(
            (self.analog_pwr_squelch_xx_0, 0), (self.blocks_complex_to_mag_squared_0, 0)
        )
        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect(
            (self.analog_pwr_squelch_xx_0, 0), (self.qtgui_waterfall_sink_x_0, 0)
        )
        self.connect(
            (self.blocks_complex_to_mag_squared_0, 0),
            (self.blocks_moving_average_xx_0, 0),
        )
        self.connect(
            (self.blocks_moving_average_xx_0, 0), (self.qtgui_number_sink_0, 0)
        )
        self.connect((self.osmosdr_source_0, 0), (self.analog_pwr_squelch_xx_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "birds_eye_squelch_2")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.analog_pwr_squelch_xx_0.set_threshold(self.threshold)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.osmosdr_source_0.set_sample_rate(self.samp_rate)
        self.osmosdr_source_0.set_bandwidth(self.samp_rate, 0)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)
        self.qtgui_waterfall_sink_x_0.set_frequency_range(
            self.center_freq, self.samp_rate
        )

    def get_gain_0(self):
        return self.gain_0

    def set_gain_0(self, gain_0):
        self.gain_0 = gain_0

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.osmosdr_source_0.set_gain(self.gain, 0)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.osmosdr_source_0.set_center_freq(self.center_freq, 0)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)
        self.qtgui_waterfall_sink_x_0.set_frequency_range(
            self.center_freq, self.samp_rate
        )


def main(top_block_cls=birds_eye_squelch_2, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string("qtgui", "style", "raster")
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()


if __name__ == "__main__":
    main()
