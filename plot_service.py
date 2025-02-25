from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import pacf, acf

from functions.general import DateTimeConverter
from functions.plots.name_mapping import get_name


class PlotService:
    def __init__(self, use_matplotlib=True, height=400, width=800):
        self.use_matplotlib = use_matplotlib
        self.height = height
        self.width = width
        self.colors = list(px.colors.qualitative.Set1 + px.colors.qualitative.Set3)

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

    def _init_figure(self, title):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=self.height,
            width=self.width,
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

    def plot_line_chart(self, array_or_df, x_column=None, y_columns=None, title="Line Chart", use_matplotlib=None):
        y_columns = self._determine_y_columns(array_or_df, x_column, y_columns)

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            for y_column in y_columns:
                x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
                plt.plot(x_data, y_data, label=get_name(y_column))
            plt.title(title)
            plt.legend()
            plt.show()
        else:
            fig = self._init_figure(title)
            for y_column in y_columns:
                x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=get_name(y_column)))
            fig.show()

    def plot_bar_chart(self, array_or_df, x_column=None, y_columns=None, title="Bar Chart", use_matplotlib=None):
        x_data = self._init_x_data(array_or_df, x_column)
        y_columns = y_columns or array_or_df.columns

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            for y_column in y_columns:
                plt.bar(x_data, array_or_df[y_column], label=get_name(y_column))
            plt.title(title)
            plt.legend()
            plt.show()
        else:
            fig = self._init_figure(title)
            for y_column in y_columns:
                fig.add_trace(go.Bar(x=x_data, y=array_or_df[y_column], name=get_name(y_column)))
            fig.show()

    def plot_histogram(self, array_or_df, columns=None, hue_column=None, title="Histogram", gap_size=0.2, bins=30,
                       use_matplotlib=None):
        columns = self._determine_y_columns(array_or_df, None, columns)

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            if hue_column:
                groups = array_or_df.groupby(hue_column)
                for hue, group in groups:
                    plt.hist(group[columns[0]], bins=bins, alpha=0.5, label=str(hue))
                plt.legend(title=hue_column)
            else:
                for col in columns:
                    plt.hist(array_or_df[col], bins=bins, alpha=0.5, label=get_name(col))
                plt.legend()
            plt.title(title)
            plt.show()
        else:
            fig = self._init_figure(title)
            if hue_column:
                groups = array_or_df.groupby(hue_column)
                for hue, group in groups:
                    fig.add_trace(go.Histogram(x=group[columns[0]], name=str(get_name(hue)), opacity=0.75))
                fig.update_layout(barmode='overlay', legend_title=f"{get_name(hue_column)}")
            else:
                for col in columns:
                    fig.add_trace(go.Histogram(x=array_or_df[col], name=get_name(col), opacity=0.75))
                fig.update_layout(bargap=gap_size)
            fig.show()

    def plot_acf_pacf(self, series, alpha=0.05, lags=None, title=None, b_pacf=False, use_matplotlib=None):
        lags = min(lags, int(len(series) // 2)) if lags is not None else lags

        corr_array = (pacf(series.dropna(), alpha=alpha, nlags=lags) if b_pacf
                      else acf(series.dropna(), alpha=alpha, nlags=lags))

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            plt.bar(range(len(corr_array[0])), corr_array[0], yerr=[
                corr_array[0] - corr_array[1][:, 0],
                corr_array[1][:, 1] - corr_array[0]
            ])
            plt.title(title or ('PACF' if b_pacf else 'ACF'))
            plt.show()
        else:
            lower_y = corr_array[1][:, 0] - corr_array[0]
            upper_y = corr_array[1][:, 1] - corr_array[0]

            fig = go.Figure()
            [fig.add_scatter(x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='#3f3f3f')
             for x in range(len(corr_array[0]))]
            fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                            marker_size=12)
            fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
            fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',
                            fillcolor='rgba(32, 146, 230,0.3)',
                            fill='tonexty', line_color='rgba(255,255,255,0)')
            fig.update_traces(showlegend=False)
            fig.update_layout(title=title or ('PACF' if b_pacf else 'ACF'))
            fig.show()

    def plot_boxplot(self, array_or_df, columns=None, title="Boxplot", use_matplotlib=None):
        columns = columns or array_or_df.columns

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            data = [array_or_df[col] for col in columns]
            plt.boxplot(data, labels=[get_name(col) for col in columns])
            plt.title(title)
            plt.show()
        else:
            fig = self._init_figure(title)
            for column in columns:
                fig.add_trace(go.Box(y=array_or_df[column], name=get_name(column)))
            fig.show()

    def plot_scatter_matrix(self, array_or_df, title="Scatter Plot Matrix", use_matplotlib=None):
        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            pd.plotting.scatter_matrix(array_or_df, figsize=(15, 15), alpha=0.5)
            plt.suptitle(title)
            plt.show()
        else:
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
                            go.Scattergl(
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
        'F': np.random.binomial(n=100, p=0.3, size=num_entries),  # Binomialverteilung
        'Category': np.random.choice(['Group1', 'Group2', 'Group3'], size=num_entries)  # Kategorien
    }

    df = pd.DataFrame(data)

    ps = PlotService()
    ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot")
    ps.plot_bar_chart(df.head(10), y_columns=['A', 'C'], title="Bar Plot")
    ps.plot_histogram(df, columns=['A', 'B'], title="Histogram")
    ps.plot_histogram(df, columns=['A'], hue_column='Category', title="Histogram with Hue")
    ps.plot_boxplot(df, columns=['A', 'C'], title="Boxplot")
    ps.plot_acf_pacf(df['A'], b_pacf=True)
    ps.plot_acf_pacf(df['A'])
    ps.plot_scatter_matrix(df, title="Scatter Plot Matrix")

    ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot", use_matplotlib=True)
    ps.plot_bar_chart(df.head(10), y_columns=['A', 'C'], title="Bar Plot", use_matplotlib=True)
    ps.plot_histogram(df, columns=['A', 'B'], title="Histogram", use_matplotlib=True)
    ps.plot_histogram(df, columns=['A'], hue_column='Category', title="Histogram with Hue", use_matplotlib=True)
    ps.plot_boxplot(df, columns=['A', 'C'], title="Boxplot", use_matplotlib=True)
    ps.plot_acf_pacf(df['A'], b_pacf=True, use_matplotlib=True)
    ps.plot_acf_pacf(df['A'], use_matplotlib=True)
    ps.plot_scatter_matrix(df, title="Scatter Plot Matrix", use_matplotlib=True)
