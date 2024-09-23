from ydata_profiling import ProfileReport
import pandas as pd
profile = ProfileReport(pd.read_csv('files/clean.csv'), title="Profiling Report")
profile.to_file("your_report.html")
