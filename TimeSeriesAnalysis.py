import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import logging
import functools
import time

LOG_FORMAT = '%(levelname)s %(asctime)s - %(message)s'

matplotlib.rcParams['backend'] = 'Qt5Agg'


class TimeSeriesAnalysis:
    """
    Given a .csv file with two columns of data, this module will auto generate and log:
        mean/median/mode
        Bin range values for a bin set of "[0-10]"

    By calling on the below methods, it will also generate and show/save graphs:
        .histogram()
        .line_graph_raw()
        .line_graph_raw_diff()
        .line_graph_rms()

    And it will log the time taken to generate each graph.
    """
    def __init__(self, filename='default'):
        self.row_count = 0
        self.y_min_value = 0
        self.y_max_value = 0
        self.x_title = ''
        self.y_title = ''
        self.filename = filename

        if filename != 'default':
            self.data = pd.read_csv(filename)

        self.mean_y = 0
        self.mode_y = 0
        self.median_y = 0
        self.bins = []
        self.bin_range = 0

        self.logger = logging.getLogger()
        self.build_logger()

        if len(self.data.columns) != 2:
            self.logger.warning('The supplied data has fewer or more than two columns.')
            self.logger.warning('Please ensure your X and Y data are in the first and second columns')

        # Sample rate for the mean in RMS functions
        self.sample_rate = int(len(self.data) / 30)

        self.log_data_header()
        self.calc_constants()
        self.bins_list()

    def build_logger(self):
        logging.basicConfig(filename='TimeSeriesLogger.log',
                            level=logging.INFO,
                            format=LOG_FORMAT,
                            filemode='w')

        self.logger.info('########################################################')
        self.logger.info('########                              ##################')
        self.logger.info('########        Running New File      ##################')
        self.logger.info('########                              ##################')
        self.logger.info('########################################################')

    def log_data_header(self):
        self.logger.info('---------- Header ----------')
        self.logger.info(f'\n{self.data}')

    def calc_constants(self):
        start_time = time.time()

        self.logger.info('---------- Constants ----------')

        # Init dataframe with csv data
        df = self.data

        # Column Titles
        columns = list(df.columns)
        self.x_title, self.y_title = columns[0], columns[1]

        self.logger.info(f'Columns: {self.x_title} - {self.y_title}')

        # Row Count
        self.row_count, _ = df.shape
        self.logger.info(f'Row Count: {self.row_count}')

        # y_max, y_min
        self.y_max_value = df[self.y_title].max()
        self.y_min_value = df[self.y_title].min()
        self.logger.info(f'Y Max value: {self.y_max_value}')
        self.logger.info(f'Y Max value: {self.y_min_value}')


        # mean, median, mode
        self.logger.info(' -------------- MMM ---------------')
        mode_y_series = df[self.y_title].mode()
        self.mode_y = float(mode_y_series.values)
        self.mean_y = df[self.y_title].mean()
        self.median_y = df[self.y_title].median()
        self.logger.info(f'Mean: {self.mean_y}, Median: {self.median_y}, Mode: {self.mode_y}')

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Constants calculated in {elapsed_time:.4f}s')

    def bins_list(self):
        start_time = time.time()
        bin_range_list = [0 for _ in range(11)]
        bin_range = (float(self.y_max_value) - float(self.y_min_value)) / 10
        y_new_floor = float(self.y_min_value)
        for i in range(len(bin_range_list)):
            bin_range_list[i] = y_new_floor
            y_new_floor = y_new_floor + float(bin_range)
        self.bins = bin_range_list
        self.bin_range = bin_range

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info('---------- Bins List ----------')
        self.logger.info(self.bins)
        self.logger.info(f'Bins calculated in {elapsed_time:.4f}s')

    def histogram(self, show=''):
        """
        Generate and save a histogram based on the given .csv

        If you can display matplotlib GUI - show=True will display interactive data
        Else, default is false
        """

        start_time = time.time()
        plt.clf()
        plt.style.use('fivethirtyeight')

        y_values = self.data[self.y_title]

        plt.hist(y_values, bins=self.bins, edgecolor='black', log=True)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.xticks(self.bins, rotation=45)
        plt.title(f'Histogram for: {self.filename}', fontsize=15)

        if 'volt' in str(self.filename).lower():
            plt.xlabel(f'Voltage (V) by Bins [0-10] - Bin Range {self.bin_range} ')
        elif 'current' in str(self.filename).lower():
            plt.xlabel(f'Current (A) by Bins [0-10] - Bin Range {self.bin_range} ')
        else:
            plt.xlabel(f'Unknown Unit by Bins [0-10] - Bin Range {self.bin_range} ')
        plt.ylabel('Frequency')

        plt.tight_layout()

        if show:
            plt.show()
        plt.savefig(f'results/{self.filename}_histogram.png')

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Histogram created in {elapsed_time:.4f}s')

    def line_graph_raw(self, show=''):
        """
        Generate and save a line graph based on the raw data in the given .csv

        If you can display matplotlib GUI - show=True will display interactive data
        Else, default is false
        """
        start_time = time.time()

        plt.clf()
        plt.style.use('fivethirtyeight')


        # init starting dataframe
        df = self.data


        plt.xticks(rotation=45)
        plt.plot(df[self.x_title],df[self.y_title], linewidth=0.5)

        plt.title(f'Raw Line Graph for: {self.filename}', fontsize=15)
        plt.xlabel(f'Time (s)')
        if 'volt' in str(self.filename).lower():
            plt.ylabel('Volts (V)')
        elif 'current' in str(self.filename).lower():
            plt.ylabel('Current (Amps)')
        else:
            plt.ylabel('Y-Value (unit unknown)')

        plt.tight_layout()

        if show:
            plt.show()

        plt.savefig(f'results/{self.filename}_linegraph_raw.png')

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Raw Line Graph created in {elapsed_time:.4f}s')

    def line_graph_raw_diff(self, show=''):
        """
        Generate and save a line graph based on the difference between values in the
            second column (y-data).

        If you can display matplotlib GUI - show=True will display interactive data
        Else, default is false
        """
        start_time = time.time()
        plt.clf()
        plt.style.use('fivethirtyeight')

        # init starting dataframe
        df = self.data

        # init diff recipient df
        diff_data = pd.DataFrame()

        # create the df for iteration
        # Duplicate the y_values column, then shift the duplicate
        selected_columns = df[[self.y_title, self.y_title]]
        new_df = selected_columns.copy()
        new_df.columns = ['a', 'b']
        new_df['b'] = new_df['b'].shift(-1)
        new_df = new_df.dropna()

        # Perform vectorized operation
        # Vector operations are performed on entire arrays, as opposed to iterating through series
        diff_data['Diff.'] = new_df['b'] - new_df['a']

        plt.xticks(rotation=45)
        plt.plot(diff_data, linewidth=.08)
        plt.title(f'Change in Y for: {self.filename}', fontsize=15)
        plt.xlabel(f'Time (s)')

        if 'volt' in str(self.filename).lower():
            plt.ylabel('Change in Volts (V)')
        elif 'current' in str(self.filename).lower():
            plt.ylabel('Change in Current (Amps)')
        else:
            plt.ylabel('Change in Y-Value (unit unknown)')

        plt.tight_layout()

        if show:
            plt.show()
        plt.savefig(f'results/{self.filename}_linegraph_raw_diff.png')

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Line Graph of Difference in Y created in {elapsed_time:.4f}s')

    def line_graph_rms(self, show=''):
        """
        Generate and save a line graph based on the
        calculated RMS of each point of data in the .csv.

        The Mean sampling rate is set at default as len(df) / 30

        If you can display matplotlib GUI - show=True will display interactive data
        Else, default is false
        """

        start_time = time.time()

        plt.clf()
        plt.style.use('fivethirtyeight')

        # init starting dataframe
        df = self.data

        # Generate RMS data
        df[self.y_title] = df[self.y_title].apply(np.square)
        df[self.y_title] = df[self.y_title].rolling(window=self.sample_rate, closed='neither', min_periods=1).mean()
        df[self.y_title] = df[self.y_title].apply(np.sqrt)

        plt.xticks(rotation=45)
        plt.plot(df[self.x_title], df[self.y_title], linewidth=1)
        plt.title(f'Root Mean Square Line Graph for: {self.filename}', fontsize=15)
        plt.xlabel(f'time in (s)')


        if 'volt' in str(self.filename).lower():
            plt.ylabel('V_RMS (V) test')
        elif 'current' in str(self.filename).lower():
            plt.ylabel('I_RMS (A)')
        else:
            plt.ylabel('RMS in Y-Value (unit unknown)')

        plt.tight_layout()

        if show:
            plt.show()

        plt.savefig(f'results/{self.filename}_linegraph_rms.png')

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f'Line Graph of RMS created in {elapsed_time:.4f}s')



