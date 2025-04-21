import numpy as np
import matplotlib.dates as md
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
import statsmodels.api as sm
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
    def _init_x_data(array_or_df, x_column, map_names=False):
        x_data = array_or_df.index if x_column is None else array_or_df[x_column]
        if isinstance(x_data, np.ndarray):
            is_datetime = np.issubdtype(x_data.dtype, np.datetime64)
        else:
            is_datetime = pd.api.types.is_datetime64_any_dtype(x_data)
        if is_datetime:
            datetime_converter = DateTimeConverter()
            x_data = datetime_converter.convert_series_to_local_datetime(x_data)

        if x_column is not None and map_names:
            array_or_df = array_or_df.copy()
            x_data = array_or_df[x_column].map(get_name).fillna(array_or_df[x_column])

        return x_data

    @staticmethod
    def _determine_y_columns(array_or_df, x_column, y_columns):
        if not isinstance(y_columns, list) and y_columns is not None:
            y_columns = [y_columns]

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

    def _plot_line_chart_matplotlib(self, array_or_df, x_column, y_columns, title, multi_axes, hline):
        if multi_axes and len(y_columns) > 1:
            fig, ax1 = plt.subplots(figsize=(self.width / 100, self.height / 100))
            axes = [ax1]

            num_axes = min(len(y_columns), 4)

            axis_mapping = {}
            for i, cols in enumerate(y_columns[:num_axes]):
                y_label = ', '.join(get_name(col) for col in cols)
                if i == 0:
                    ax1.set_ylabel(y_label)
                    ax = ax1
                else:
                    ax = ax1.twinx()
                    ax.set_ylabel(y_label)
                    ax.spines[f'right' if i % 2 else 'left'].set_position(("outward", int(i / 2) * 50))
                    axes.append(ax)

                for col in cols:
                    axis_mapping[col] = ax

            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i, y_column in enumerate(axis_mapping.keys()):
                x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
                if pd.api.types.is_datetime64_any_dtype(x_data):
                    axis_mapping[y_column].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
                axis_mapping[y_column].plot(x_data, y_data, label=get_name(y_column), color=colors[i % len(colors)])

            handles, labels = zip(*[ax.get_legend_handles_labels() for ax in axes])
            handles, labels = sum(handles, []), sum(labels, [])
            plt.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=len(y_columns))
        else:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            for y_column in y_columns:
                x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
                plt.plot(x_data, y_data, label=get_name(y_column))

                plt.ylabel(get_name(y_column))

                if pd.api.types.is_datetime64_any_dtype(x_data):
                    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

                if pd.api.types.is_integer_dtype(x_data):
                    plt.xticks(np.unique(x_data))

                plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=len(y_columns))

        if isinstance(hline, (int, float)):
            plt.axhline(hline, color='gray', linestyle=':')

        plt.xlabel(get_name(x_column))
        plt.title(title)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.show()

    def _plot_line_chart_plotly(self, array_or_df, x_column, y_columns, title, multi_axes):
        fig = self._init_figure(title)

        if multi_axes and len(y_columns) > 1:
            num_axes = min(len(y_columns), 4)
            axis_mapping = {col: f"y{i + 1}" for i, cols in enumerate(y_columns[:num_axes]) for col in cols}

            for y_column in y_columns:
                x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=get_name(y_column),
                                         yaxis=axis_mapping[y_column]))

            layout_updates = {
                "yaxis1": dict(side="left", showgrid=True)
            }
            for i in range(1, num_axes):
                layout_updates[f"yaxis{i + 1}"] = dict(
                    overlaying="y",
                    side="right" if i % 2 else "left",
                    showgrid=False,
                    showticklabels=True
                )

        else:
            for y_column in y_columns:
                x_data, y_data = self._init_xy_data(array_or_df, x_column, y_column)
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=get_name(y_column)))

            layout_updates = {
                "yaxis": dict(side="left", showgrid=True)
            }

        fig.update_layout(**layout_updates)
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5))
        fig.show()

    def plot_line_chart(self, array_or_df, x_column=None, y_columns=None, title="Line Chart", multi_axes=False,
                        hline=None, use_matplotlib=None):
        y_columns = self._determine_y_columns(array_or_df, x_column, y_columns)

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            self._plot_line_chart_matplotlib(array_or_df, x_column, y_columns, title, multi_axes, hline)
        else:
            self._plot_line_chart_plotly(array_or_df, x_column, y_columns, title, multi_axes)

    def plot_bar_chart(self, array_or_df, x_column=None, y_columns=None, title="Bar Chart", horizontal=False,
                       use_matplotlib=None):
        x_data = self._init_x_data(array_or_df, x_column, map_names=True)
        y_columns = self._determine_y_columns(array_or_df, x_column, y_columns)

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            for y_column in y_columns:
                y_data = array_or_df[y_column]

                if len(y_columns) == 1 and np.issubdtype(y_data.dtype, np.number):
                    colors = ['orange' if v < 0 else 'steelblue' for v in y_data]
                else:
                    colors = None

                if horizontal:
                    plt.barh(x_data, array_or_df[y_column], color=colors)
                else:
                    plt.bar(x_data, array_or_df[y_column], color=colors)

            if horizontal:
                plt.ylabel(get_name(x_column))
                if len(y_columns) == 1:
                    plt.xlabel(get_name(y_columns[0]))
            else:
                plt.xlabel(get_name(x_column))
                if len(y_columns) == 1:
                    plt.ylabel(get_name(y_columns[0]))

            plt.xticks(rotation=45, ha='right')
            plt.title(title)
            plt.tight_layout()
            plt.show()
        else:
            fig = self._init_figure(title)
            for y_column in y_columns:
                fig.add_trace(go.Bar(x=x_data, y=array_or_df[y_column], name=get_name(y_column)))
            fig.show()

    def plot_histogram(self, array_or_df, columns=None, hue_column=None, title="Histogram", gap_size=0.2, bins=30,
                       scale_y=False, use_matplotlib=None):
        columns = self._determine_y_columns(array_or_df, None, columns)

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            if hue_column:
                groups = array_or_df.groupby(hue_column)
                for hue, group in groups:
                    plt.hist(group[columns[0]], bins=bins, alpha=0.6, label=str(hue))
                plt.legend(title=hue_column)
            else:
                for col in columns:
                    plt.hist(array_or_df[col], bins=bins, alpha=0.6, label=get_name(col))
                plt.legend()

            if scale_y:
                plt.yscale('log')

            plt.ylabel('Anzahl')
            plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
            plt.title(title)
            plt.tight_layout()
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

    def plot_heatmap(self, array_or_df, title="Heatmap", use_matplotlib=None):
        corr = array_or_df.copy()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        masked_corr = corr.mask(mask)
        non_empty_rows = ~masked_corr.isnull().all(axis=1)
        non_empty_cols = ~masked_corr.isnull().all(axis=0)

        corr = corr.loc[non_empty_rows, non_empty_cols]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        n_features = len(corr.columns)
        cell_size = 0.5
        width = max(6.0, n_features * cell_size)
        height = max(4.0, len(corr) * cell_size)

        font = 6

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(width, height))

            cmap = sns.diverging_palette(230, 20, as_cmap=True)

            ax = sns.heatmap(
                corr,
                mask=mask,
                cmap=cmap,
                vmax=1,
                center=0.5,
                annot=True,
                fmt=".2f",
                annot_kws={"size": font},
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5}
            )

            ax.set_xticklabels(
                [get_name(label.get_text()) for label in ax.get_xticklabels()], fontsize=font)
            ax.set_yticklabels(
                [get_name(label.get_text()) for label in ax.get_yticklabels()], fontsize=font)

            plt.yticks(rotation=0)
            plt.xticks(rotation=45, ha='right')

            plt.title(title)
            plt.tight_layout()
            plt.show()

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
            plt.tight_layout()
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

    def plot_boxplot(self, array_or_df, column_x, column_y, title=None, scale_y=False, use_matplotlib=None):
        label_x = get_name(column_x)
        label_y = get_name(column_y)

        if isinstance(array_or_df, pd.DataFrame):
            df = array_or_df[[column_x, column_y]].dropna()

            grouped = df.groupby(column_x)[column_y].apply(list)
            labels = grouped.index.tolist()
            values = grouped.values

        elif isinstance(array_or_df, np.ndarray) and array_or_df.dtype.names:
            mask = ~np.isnan(array_or_df[column_y])
            data = array_or_df[mask]

            unique_keys = np.unique(data[column_x])
            labels = unique_keys.tolist()
            values = [data[column_y][data[column_x] == key] for key in unique_keys]
        else:
            raise TypeError("Daten müssen ein Pandas DataFrame oder strukturiertes NumPy-Array sein.")

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))
            plt.boxplot(values)

            if scale_y:
                plt.yscale('log')

            plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=45)
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.title(title or f'Boxplot: {label_y} pro {label_x}')
            plt.tight_layout()
            plt.show()
        else:
            fig = self._init_figure(title or f'Boxplot: {label_y} pro {label_x}')

            for label, val_list in zip(labels, values):
                fig.add_trace(go.Box(
                    y=val_list,
                    name=str(label),
                    boxpoints='outliers',
                    marker=dict(opacity=0.7)
                ))

            fig.update_layout(
                xaxis_title=label_x,
                yaxis_title=label_y,
                yaxis_type='log' if scale_y else 'linear',
                margin=dict(t=50, b=80, l=80, r=20)
            )
            fig.show()

    def plot_scatter_plot(self, array_or_df, x_column=None, y_column=None, title="Scatter Plot",
                          hue_column=None, trend=False, use_matplotlib=None):
        x_data = self._init_x_data(array_or_df, x_column)
        y_data = array_or_df[y_column]

        trendline = np.empty((0, 2))
        if trend:
            lowess = sm.nonparametric.lowess
            trendline = lowess(y_data, x_data, frac=0.3)

        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            plt.figure(figsize=(self.width / 100, self.height / 100))

            sns.scatterplot(data=array_or_df, x=x_column, y=y_column, alpha=0.7, s=5,
                            hue=array_or_df[hue_column].map({0: "Stadt", 1: "Überland"}),
                            hue_order=['Stadt', 'Überland'])

            if trend:
                plt.plot(trendline[:, 0], trendline[:, 1], color='red', label='Trendlinie (LOWESS)')
                plt.axhline(0, color='black', linestyle='dashed')

            plt.xlabel(get_name(x_column))
            plt.ylabel(get_name(y_column))
            plt.title(title)
            if plt.gca().get_legend_handles_labels()[1]:
                plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            fig = self._init_figure(title)
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name=get_name(y_column)))
            if trend:
                fig.add_trace(go.Scatter(x=trendline[:, 0], y=trendline[:, 1], mode='lines', name='Trendlinie',
                                         line=dict(color='red')))
                fig.add_hline(y=0, line_dash='dash', line_color='gray', name='y = 0')
            fig.show()

    def plot_scatter_matrix(self, array_or_df, title="Scatter Plot Matrix", figsize=(12, 12), nbinsx=50,
                            use_matplotlib=None):
        if use_matplotlib if isinstance(use_matplotlib, bool) else self.use_matplotlib:
            sns.pairplot(array_or_df, plot_kws={'s': 5, 'alpha': 0.3})

            for ax in plt.gcf().axes:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

            plt.suptitle(title)
            plt.tight_layout()
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
                                nbinsx=nbinsx,
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
    data['G'] = data['A'] ** 2 + np.random.normal(loc=0, scale=0.5, size=num_entries)

    df = pd.DataFrame(data)

    ps = PlotService(use_matplotlib=False)
    # ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot")
    # ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot", multi_axes=True)
    # ps.plot_bar_chart(df.head(10), y_columns=['A', 'C'], title="Bar Plot")
    # ps.plot_histogram(df, columns=['A', 'B'], title="Histogram")
    # ps.plot_histogram(df, columns=['A'], hue_column='Category', title="Histogram with Hue")
    # ps.plot_boxplot(df, column_x='Category', column_y='A')
    # ps.plot_acf_pacf(df['A'], b_pacf=True)
    # ps.plot_acf_pacf(df['A'])
    # ps.plot_scatter_matrix(df, title="Scatter Plot Matrix")
    # ps.plot_scatter_plot(df, x_column='A', y_column='G', title="Scatter Plot")
    # ps.plot_scatter_plot(df, x_column='A', y_column='G', title="Scatter Plot", trend=True)

    # ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot", use_matplotlib=True)
    # ps.plot_line_chart(df, y_columns=['A', 'B'], title="Line Plot", multi_axes=True, use_matplotlib=True)
    # ps.plot_bar_chart(df.head(10), y_columns=['A', 'C'], title="Bar Plot", use_matplotlib=True)
    # ps.plot_histogram(df, columns=['A', 'B'], title="Histogram", use_matplotlib=True)
    # ps.plot_histogram(df, columns=['A'], hue_column='Category', title="Histogram with Hue", use_matplotlib=True)
    # ps.plot_boxplot(df, column_x='Category', column_y='A', use_matplotlib=True)
    # ps.plot_acf_pacf(df['A'], b_pacf=True, use_matplotlib=True)
    # ps.plot_acf_pacf(df['A'], use_matplotlib=True)
    # ps.plot_scatter_matrix(df, title="Scatter Plot Matrix", use_matplotlib=True)
    # ps.plot_scatter_plot(df, x_column='A', y_column='G', title="Scatter Plot", use_matplotlib=True)
    # ps.plot_scatter_plot(df, x_column='A', y_column='G', title="Scatter Plot", trend=True, use_matplotlib=True)
    # ps.plot_scatter_plot(df, x_column='A', y_column='G', title="Scatter Plot", hue_column='Category',
    #                      use_matplotlib=True)
    ps.plot_heatmap(df[['A', 'B', 'C', 'D', 'E', 'F']].corr(), use_matplotlib=True)
