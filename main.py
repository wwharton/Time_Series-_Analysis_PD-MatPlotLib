from TimeSeriesAnalysis import TimeSeriesAnalysis as tsa




if __name__ == '__main__':
    filename = 'Pack_Volt_Data.csv'
    # filename = 'Current_Data.csv'
    tsa = tsa(filename)

    # Write doc strings for methods to call, what do they dp
    tsa.histogram()
    tsa.line_graph_raw()
    tsa.line_graph_raw_diff()
    tsa.line_graph_rms(show=True)



