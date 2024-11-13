import numpy as np
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, acf

from functions.general import DateTimeConverter
from functions.plots.name_mapping import get_name


class PlotService:
    def __init__(self):
        pass

    @staticmethod
    def _init_figure(title):
        """Initializes a Plotly figure with standard layout options."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis=dict(tickmode="auto"),
            hovermode="x unified",
            modebar_add=["zoom", "pan", "resetScale2d", "select2d", "lasso2d"],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=-0.2,
                x=0.5,
                xanchor="center",
                orientation="h"
            )
        )
        return fig

    @staticmethod
    def _init_x_data(array_or_df, x_column):
        x_data = array_or_df.index if x_column is None else array_or_df[x_column]
        if isinstance(x_data, np.ndarray):
            is_datetime = np.issubdtype(x_data.dtype, np.datetime64)
        else:
            is_datetime = pd.api.types.is_datetime64_any_dtype(x_data)
        if is_datetime:
            datetime_converter = DateTimeConverter()
            x_data = datetime_converter.convert_series_to_local_datetime(x_data)
        return x_data

    @staticmethod
    def _determine_y_columns(array_or_df, x_column, y_columns):
        if isinstance(array_or_df, pd.DataFrame):
            if x_column is None:
                return y_columns or list(array_or_df.columns)
            else:
                return y_columns or [col for col in array_or_df.columns if col != x_column]
        elif isinstance(array_or_df, np.ndarray) and array_or_df.dtype.names:
            if x_column is None:
                raise ValueError("Structured NumPy arrays require an explicit x_column.")
            else:
                return y_columns or [name for name in array_or_df.dtype.names if name != x_column]
        else:
            raise TypeError("Input must be a DataFrame or a structured NumPy array with named columns")

    def _init_xy_data(self, array_or_df, x_column, y_column):
        if isinstance(array_or_df, pd.DataFrame):
            mask = array_or_df[y_column].notna()
        else:
            mask = (~np.isnan(array_or_df[y_column])) & (array_or_df[y_column] is not None)
        x_data_filtered = self._init_x_data(array_or_df[mask], x_column)
        y_data_filtered = array_or_df[y_column][mask]

        if x_column is None:
            y_data_filtered = y_data_filtered.reindex(x_data_filtered)

        return x_data_filtered, y_data_filtered

    def plot_line_chart(self, array_or_df, x_column=None, y_columns=None, title="Line Chart"):
        """Plots a line chart using the provided DataFrame."""
        fig = self._init_figure(title)
        y_columns = self._determine_y_columns(array_or_df, x_column, y_columns)

        for y_column in y_columns:
            x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines',
                name=get_name(y_column)
            ))
        fig.show()

    def plot_bar_chart(self, array_or_df, x_column=None, y_columns=None, title="Bar Chart"):
        """Plots a bar chart using the provided DataFrame."""
        fig = self._init_figure(title)

        x_data = self._init_x_data(array_or_df, x_column)
        y_columns = y_columns or array_or_df.columns

        for y_column in y_columns:
            fig.add_trace(go.Bar(
                x=x_data,
                y=array_or_df[y_column],
                name=get_name(y_column)
            ))
        fig.show()

    def plot_histogram(self, array_or_df, columns=None, title="Histogram", gap_size=0.2):
        """Plots a histogram for the selected columns."""
        fig = self._init_figure(title)

        columns = columns or array_or_df.columns

        for column in columns:
            fig.add_trace(go.Histogram(
                x=array_or_df[column],
                name=get_name(column),
                opacity=0.75,
            ))
        fig.update_layout(bargap=gap_size)
        fig.show()

    def plot_acf_pacf(self, series, alpha=0.05, lags=None, title=None, b_pacf=False):
        """Plots ACF and PACF for a given time series."""
        lags = min(lags, int(len(series) // 2)) if lags is not None else lags

        corr_array = (pacf(series.dropna(), alpha=alpha, nlags=lags) if b_pacf
                      else acf(series.dropna(), alpha=alpha, nlags=lags))
        lower_y = corr_array[1][:, 0] - corr_array[0]
        upper_y = corr_array[1][:, 1] - corr_array[0]

        fig = go.Figure()
        [fig.add_scatter(x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='#3f3f3f')
         for x in range(len(corr_array[0]))]
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                        marker_size=12)
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
        fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines', fillcolor='rgba(32, 146, 230,0.3)',
                        fill='tonexty', line_color='rgba(255,255,255,0)')
        fig.update_traces(showlegend=False)
        fig.update_xaxes(range=[-1, 42])
        fig.update_yaxes(zerolinecolor='#000000')

        title = title or ('Partial Autocorrelation (PACF)' if b_pacf else 'Autocorrelation (ACF)')
        fig.update_layout(title=title)
        fig.show()

    def plot_boxplot(self, array_or_df, columns=None, title="Boxplot"):
        """Plots a boxplot for the selected columns."""
        fig = self._init_figure(title)

        columns = columns or array_or_df.columns

        for column in columns:
            fig.add_trace(go.Box(
                y=array_or_df[column],
                name=get_name(column)
            ))
        fig.show()

    @staticmethod
    def plot_scatter_matrix(array_or_df, title="Scatter Plot Matrix"):
        """
        Plots a scatter plot matrix showing the correlation between each pair of columns.

        Args:
            array_or_df (DataFrame or NumPy structured array): The data to plot, with named columns.
            title (str): The title of the plot.

        Returns:
            None
        """
        if isinstance(array_or_df, np.ndarray):
            array_or_df = pd.DataFrame(array_or_df)

        columns = array_or_df.columns
        num_columns = len(columns)

        fig = make_subplots(rows=num_columns, cols=num_columns,
                            subplot_titles=[f"{col1} vs {col2}" for col1 in columns for col2 in columns])

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    fig.add_trace(
                        go.Scatter(
                            x=array_or_df[col1],
                            y=array_or_df[col2],
                            mode='markers',
                            marker=dict(opacity=0.6),
                            showlegend=False
                        ),
                        row=i + 1, col=j + 1
                    )
                else:
                    fig.add_trace(
                        go.Histogram(
                            x=array_or_df[col1],
                            opacity=0.75,
                            showlegend=False
                        ),
                        row=i + 1, col=j + 1
                    )

        fig.update_layout(
            title=title,
            height=300 * num_columns,
            width=300 * num_columns,
            showlegend=False
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.show()


if __name__ == "__main__":
    np.random.seed(42)
    num_entries = 1000
    data = {
        'A': np.random.normal(loc=0, scale=1, size=num_entries),  # Normalverteilung
        'B': np.random.normal(loc=5, scale=2, size=num_entries),  # Verschobene Normalverteilung
        'C': np.random.exponential(scale=1, size=num_entries),  # Exponentielle Verteilung
        'D': np.random.uniform(low=0, high=10, size=num_entries),  # Gleichverteilung
        'E': np.random.lognormal(mean=1, sigma=0.5, size=num_entries),  # Log-Normalverteilung
        'F': np.random.binomial(n=100, p=0.3, size=num_entries)  # Binomialverteilung
    }

    df = pd.DataFrame(data)

    ps = PlotService()
    ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot")
    ps.plot_bar_chart(df.head(10), y_columns=['A', 'C'], title="Bar Plot")
    ps.plot_histogram(df, columns=['A', 'B'], title="Histogram")
    ps.plot_boxplot(df, columns=['A', 'C'], title="Boxplot")
    ps.plot_acf_pacf(df['A'], b_pacf=True)
    ps.plot_acf_pacf(df['A'])
    ps.plot_scatter_matrix(df, title="Scatter Plot Matrix")
