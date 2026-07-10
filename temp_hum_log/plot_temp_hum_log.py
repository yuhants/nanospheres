"""Plot temperature/humidity logs from a Thorlabs TSP01B CSV export.

Usage:
    python plot_temp_hum_log.py <path_to_csv>
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_log(path):
    df = pd.read_csv(path, sep=';', skiprows=5, encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], format='%b %d %Y %H:%M:%S'
    )
    for col in ['Temperature[\xb0C]', 'Humidity[%]', 'TH1[\xb0C]', 'TH2[\xb0C]']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_log(df, title=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax1.plot(df['datetime'], df['Temperature[\xb0C]'], label='Ambient (main)', color='tab:red')
    if 'TH1[\xb0C]' in df.columns:
        ax1.plot(df['datetime'], df['TH1[\xb0C]'], label='TH1', color='tab:orange')
    if 'TH2[\xb0C]' in df.columns and df['TH2[\xb0C]'].notna().any():
        ax1.plot(df['datetime'], df['TH2[\xb0C]'], label='TH2', color='tab:brown')
    ax1.set_ylabel('Temperature [\xb0C]')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    ax2.plot(df['datetime'], df['Humidity[%]'], color='tab:blue')
    ax2.set_ylabel('Humidity [%]')
    ax2.set_xlabel('Time')
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '../20260706_temp_hum_logging.csv'
    df = load_log(path)
    plot_log(df, title=path)
    plt.show()
