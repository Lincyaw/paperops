from typing import Optional, Union, Any
from collections.abc import Sequence
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .templates import Template
from .config import LegendStyle, LEGEND_STYLES


DataDict = dict[str, list[Any]]


class YLimMode(Enum):
    AUTO = "auto"
    DATA_EXTEND = "data_extend"
    PERCENTAGE = "percentage"
    ZERO_EXTEND = "zero_extend"
    CUSTOM = "custom"


class PlotGenerator:
    def __init__(self, template: Template) -> None:
        """
        Initialize plot generator with a template.

        Parameters:
        -----------
        template : Template
            The template to use for styling
        """
        self.template = template

    def _prepare_plot(self, plot_type: str) -> tuple[Figure, Axes]:
        """Prepare the plot with template styling."""
        self.template.apply_style(plot_type)
        return self.template.create_figure()

    def _get_legend_style_config(
        self, legend_style: str | LegendStyle
    ) -> dict[str, Any]:
        """
        Get legend style configuration from config or custom config.

        Parameters:
        -----------
        legend_style : str or LegendStyle
            Style of the legend

        Returns:
        --------
        dict[str, Any]
            Dictionary with legend style parameters
        """
        # Get legend styles from custom config if available, otherwise use defaults
        if hasattr(self.template, "custom_config") and self.template.custom_config:
            legend_styles = self.template.custom_config.get_legend_styles()
        else:
            legend_styles = LEGEND_STYLES

        # Try to find the style in the legend styles dictionary
        # Handle both LegendStyle enum and string keys
        if isinstance(legend_style, LegendStyle) and legend_style in legend_styles:
            return legend_styles[legend_style].copy()
        elif isinstance(legend_style, str):
            # Try to convert to enum first
            try:
                style_enum = LegendStyle(legend_style)
                if style_enum in legend_styles:
                    return legend_styles[style_enum].copy()
            except ValueError:
                pass  # Fall through to default

        # Fall back to clean style
        if LegendStyle.CLEAN in legend_styles:
            return legend_styles[LegendStyle.CLEAN].copy()
        else:
            return LEGEND_STYLES[LegendStyle.CLEAN].copy()

    def _adjust_xlabel_rotation(
        self, ax: Axes, x_data: Sequence[Any], max_label_length: int = 8
    ) -> None:
        """
        Automatically adjust x-axis label rotation if labels are too long.

        Parameters:
        -----------
        ax : Axes
            The matplotlib axes object
        x_data : Sequence
            The x-axis data
        max_label_length : int
            Maximum label length before rotation (default: 8)
        """
        # Get x-axis tick labels
        tick_labels = ax.get_xticklabels()

        # Check if any label is longer than max_label_length
        needs_rotation = False
        if tick_labels:
            # Check existing labels
            for label in tick_labels:
                if len(label.get_text()) > max_label_length:
                    needs_rotation = True
                    break
        else:
            # Check the data directly if no labels are set yet
            if len(x_data) == 0:
                return  # No data to check

            max_len = max(len(str(item)) for item in x_data)
            if max_len > max_label_length:
                needs_rotation = True

        # Apply rotation if needed
        if needs_rotation:
            ax.tick_params(axis="x", rotation=45)
            # Ensure labels are aligned properly
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

    def _place_legend_intelligently(
        self,
        ax: Axes,
        legend_preference: Optional[str] = None,
        legend_outside: bool = False,
        legend_style: Union[str, LegendStyle] = LegendStyle.CLEAN,
        **legend_kwargs: Any,
    ) -> None:
        """
        Intelligently place legend to avoid overlap with plot content.

        Parameters:
        -----------
        ax : Axes
            The matplotlib axes object
        legend_preference : str, optional
            Preferred legend location ('upper right', 'lower left', etc.)
            If None, will automatically determine best location
        legend_outside : bool
            If True, place legend outside the plot area
        legend_style : str or LegendStyle
            Style of the legend background and frame
        **legend_kwargs
            Additional arguments passed to ax.legend()
        """
        if not ax.get_legend_handles_labels()[0]:  # No legend data
            return

        if legend_outside:
            # Place legend outside the plot area
            legend_kwargs.setdefault("bbox_to_anchor", (1.05, 1))
            legend_kwargs.setdefault("loc", "upper left")

            # Apply legend style
            style_config = self._get_legend_style_config(legend_style)
            legend_kwargs.update(style_config)

            ax.legend(**legend_kwargs)
            return

        if legend_preference:
            # Use user-specified location
            style_config = self._get_legend_style_config(legend_style)
            legend_kwargs.update(style_config)
            ax.legend(loc=legend_preference, **legend_kwargs)
            return

        # Temporarily use simple default positioning
        # TODO: Re-implement intelligent positioning with better type safety
        best_location = "upper right"

        # Apply legend style
        style_config = self._get_legend_style_config(legend_style)
        legend_kwargs.update(style_config)

        ax.legend(loc=best_location, **legend_kwargs)

    def _find_best_legend_location(self, ax: Axes) -> str:
        """
        Find the best legend location by analyzing plot content density.

        Parameters:
        -----------
        ax : Axes
            The matplotlib axes object

        Returns:
        --------
        str
            Best legend location string
        """
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Define corner regions (in relative coordinates)
        corners = {
            "upper right": (0.7, 1.0, 0.3, 0.3),  # (x_start, y_start, width, height)
            "upper left": (0.0, 1.0, 0.3, 0.3),
            "lower right": (0.7, 0.0, 0.3, 0.3),
            "lower left": (0.0, 0.0, 0.3, 0.3),
            "center right": (0.7, 0.35, 0.3, 0.3),
            "center left": (0.0, 0.35, 0.3, 0.3),
            "upper center": (0.35, 0.85, 0.3, 0.15),
            "lower center": (0.35, 0.0, 0.3, 0.15),
        }

        # Convert relative coordinates to data coordinates
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        corner_scores = {}

        for location, (rel_x, rel_y, rel_w, rel_h) in corners.items():
            # Convert to data coordinates
            x_start = xlim[0] + rel_x * x_range
            x_end = x_start + rel_w * x_range
            y_start = ylim[0] + rel_y * y_range
            y_end = y_start + rel_h * y_range

            # Calculate density score for this region
            density = self._calculate_region_density(ax, x_start, x_end, y_start, y_end)
            corner_scores[location] = density

        # Return location with lowest density (least crowded)
        best_location = min(corner_scores.keys(), key=lambda k: corner_scores[k])

        # Fallback to common good locations if all areas are crowded
        fallback_order = ["upper right", "upper left", "lower right", "lower left"]

        if corner_scores[best_location] > 0.7:  # Very crowded
            for fallback in fallback_order:
                if corner_scores[fallback] < corner_scores[best_location]:
                    return fallback

        return best_location

    def _calculate_region_density(
        self, ax: Axes, x_start: float, x_end: float, y_start: float, y_end: float
    ) -> float:
        """
        Calculate the density of plot elements in a given region.

        Parameters:
        -----------
        ax : Axes
            The matplotlib axes object
        x_start, x_end, y_start, y_end : float
            Region boundaries in data coordinates

        Returns:
        --------
        float
            Density score (0.0 = empty, 1.0 = very crowded)
        """
        # Simplified approach to avoid complex matplotlib typing issues
        # Return a default low density score for legend placement
        return 0.3

    def _set_ylim_intelligently(
        self,
        ax: Axes,
        data: DataDict,
        y_columns: Union[str, list[str]],
        ylim_mode: Union[str, YLimMode] = "auto",
        ylim: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        Intelligently set y-axis limits based on the specified mode.

        Parameters:
        -----------
        ax : Axes
            The matplotlib axes object
        data : DataDict
            The data being plotted
        y_columns : str or list of str
            Column name(s) for y-axis data
        ylim_mode : str or YLimMode
            Y-axis limit mode:
            - "auto": matplotlib default (automatic)
            - "data_extend": min(data) to max(data)*1.1
            - "percentage": 0 to 1 (for percentage data)
            - "zero_extend": 0 to max(data)*1.1
            - "custom": use ylim parameter
        ylim : tuple, optional
            Custom y-axis limits (min, max) when ylim_mode="custom"
        """
        # Convert enum to string value if needed
        mode_str = ylim_mode.value if isinstance(ylim_mode, YLimMode) else ylim_mode

        if mode_str == "auto":
            # Let matplotlib handle it automatically
            return

        if mode_str == "custom" and ylim is not None:
            ax.set_ylim(ylim)
            return

        # Ensure y_columns is a list
        if isinstance(y_columns, str):
            y_columns = [y_columns]

        # Get all y-axis data
        y_data_all: list[float] = []
        for col in y_columns:
            if col in data:
                y_values = data[col]
                # Convert to float and filter out non-numeric values
                for val in y_values:
                    try:
                        y_data_all.append(float(val))
                    except (ValueError, TypeError):
                        continue  # Skip non-numeric values

        if not y_data_all:
            return  # No valid data found

        y_min = min(y_data_all)
        y_max = max(y_data_all)

        if mode_str == "data_extend":
            # From min to max * 1.1, with small padding on bottom
            padding = (y_max - y_min) * 0.05  # 5% padding on bottom
            ax.set_ylim(y_min - padding, y_max * 1.1)

        elif mode_str == "percentage":
            # 0 to 1 for percentage data
            ax.set_ylim(0, 1)

        elif mode_str == "zero_extend":
            # From 0 to max * 1.1
            ax.set_ylim(0, y_max * 1.1)

        else:
            # Invalid mode, fall back to auto
            return


class LinePlot(PlotGenerator):
    """Generate line plots for academic papers."""

    def create(
        self,
        data: DataDict,
        x: str,
        y: Union[str, list[str]],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = True,
        legend_preference: Optional[str] = None,
        legend_outside: bool = False,
        legend_style: Union[str, LegendStyle] = LegendStyle.CLEAN,
        ylim_mode: Union[str, YLimMode] = "auto",
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a line plot.

        Parameters:
        -----------
        data : DataDict
            Data to plot
        x : str
            Column name for x-axis
        y : str or list
            Column name(s) for y-axis
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        legend : bool
            Whether to show legend
        legend_preference : str, optional
            Specific legend location preference
        legend_outside : bool
            Whether to place legend outside plot area
        legend_style : str or LegendStyle
            Style of the legend ("clean", "gray_transparent", "white_frame", "minimal")
        ylim_mode : str or YLimMode
            Y-axis limit mode ("auto", "data_extend", "percentage", "zero_extend", "custom")
        ylim : tuple, optional
            Custom y-axis limits (min, max) when ylim_mode="custom"
        **kwargs
            Additional matplotlib arguments

        Returns:
        --------
        tuple[Figure, Axes]
            (figure, axes) objects
        """
        fig, ax = self._prepare_plot("line")

        # Get colors from template
        y_cols = [y] if isinstance(y, str) else y
        colors = self.template.get_colors(len(y_cols))

        # Plot lines
        for i, col in enumerate(y_cols):
            ax.plot(
                data[x],
                data[col],
                color=colors[i],
                label=col if legend else None,
                **kwargs,
            )

        # Set labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        # Add legend if requested and multiple lines
        if legend and len(y_cols) > 1:
            self._place_legend_intelligently(
                ax,
                legend_preference=legend_preference,
                legend_outside=legend_outside,
                legend_style=legend_style,
            )

        # Adjust x-axis label rotation if needed
        self._adjust_xlabel_rotation(ax, data[x])

        # Set y-axis limits intelligently
        self._set_ylim_intelligently(ax, data, y_cols, ylim_mode, ylim)

        plt.tight_layout()
        return fig, ax


class BarPlot(PlotGenerator):
    """Generate bar plots for academic papers."""

    def create(
        self,
        data: DataDict,
        x: str,
        y: Union[str, list[str]],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        horizontal: bool = False,
        legend: bool = True,
        legend_preference: Optional[str] = None,
        legend_outside: bool = False,
        legend_style: Union[str, LegendStyle] = LegendStyle.CLEAN,
        bar_width: float = 0.8,
        ylim_mode: Union[str, YLimMode] = "auto",
        ylim: Optional[tuple[float, float]] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a bar plot. Supports both single and grouped bar charts.

        Parameters:
        -----------
        data : DataDict
            Data to plot
        x : str
            Column name for categories
        y : str or list of str
            Column name(s) for values. If list, creates grouped bars
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        horizontal : bool
            Whether to create horizontal bars
        legend : bool
            Whether to show legend (only for grouped bars)
        legend_preference : str, optional
            Specific legend location preference
        legend_outside : bool
            Whether to place legend outside plot area
        legend_style : str or LegendStyle
            Style of the legend ("clean", "gray_transparent", "white_frame", "minimal")
        bar_width : float
            Width of bars (default: 0.8)
        ylim_mode : str or YLimMode
            Y-axis limit mode ("auto", "data_extend", "percentage", "zero_extend", "custom")
        ylim : tuple, optional
            Custom y-axis limits (min, max) when ylim_mode="custom"
        **kwargs
            Additional matplotlib arguments

        Returns:
        --------
        tuple[Figure, Axes]
            (figure, axes) objects
        """
        fig, ax = self._prepare_plot("bar")

        # Handle single or multiple y columns
        y_cols = [y] if isinstance(y, str) else y
        n_groups = len(y_cols)

        # Get colors for each group
        colors = self.template.get_colors(n_groups)

        # Get x values
        x_values = data[x]
        n_bars = len(x_values)

        if n_groups == 1:
            # Single bar plot (original behavior)
            if horizontal:
                ax.barh(x_values, data[y_cols[0]], color=colors[0], **kwargs)
            else:
                ax.bar(x_values, data[y_cols[0]], color=colors[0], **kwargs)
        else:
            # Grouped bar plot
            # Calculate bar positions
            if horizontal:
                # For horizontal bars
                bar_height = bar_width / n_groups
                y_positions = np.arange(n_bars)

                for i, col in enumerate(y_cols):
                    offset = (i - (n_groups - 1) / 2) * bar_height
                    ax.barh(
                        y_positions + offset,
                        data[col],
                        height=bar_height,
                        color=colors[i],
                        label=col if legend else None,
                        **kwargs,
                    )

                # Set y-axis ticks and labels
                ax.set_yticks(y_positions)
                ax.set_yticklabels([str(val) for val in x_values])
            else:
                # For vertical bars
                bar_width_single = bar_width / n_groups
                x_positions = np.arange(n_bars)

                for i, col in enumerate(y_cols):
                    offset = (i - (n_groups - 1) / 2) * bar_width_single
                    ax.bar(
                        x_positions + offset,
                        data[col],
                        width=bar_width_single,
                        color=colors[i],
                        label=col if legend else None,
                        **kwargs,
                    )

                # Set x-axis ticks and labels
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(val) for val in x_values])

        # Set labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        # Add legend for grouped bars
        if legend and n_groups > 1:
            self._place_legend_intelligently(
                ax,
                legend_preference=legend_preference,
                legend_outside=legend_outside,
                legend_style=legend_style,
            )

        # Adjust x-axis label rotation if needed (only for vertical bars)
        if not horizontal:
            self._adjust_xlabel_rotation(ax, x_values)

        # Set y-axis limits intelligently
        self._set_ylim_intelligently(ax, data, y_cols, ylim_mode, ylim)

        plt.tight_layout()
        return fig, ax


class ScatterPlot(PlotGenerator):
    """Generate scatter plots for academic papers."""

    def create(
        self,
        data: DataDict,
        x: str,
        y: str,
        size: Optional[str] = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a scatter plot.

        Parameters:
        -----------
        data : DataDict
            Data to plot
        x : str
            Column name for x-axis
        y : str
            Column name for y-axis
        size : str, optional
            Column name for point sizes
        color : str, optional
            Column name for point colors
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        **kwargs
            Additional matplotlib arguments

        Returns:
        --------
        tuple[Figure, Axes]
            (figure, axes) objects
        """
        fig, ax = self._prepare_plot("scatter")

        # Prepare scatter arguments
        scatter_kwargs = kwargs.copy()

        if size is not None:
            scatter_kwargs["s"] = data[size]
        if color is not None:
            scatter_kwargs["c"] = data[color]
            scatter_kwargs["cmap"] = "viridis"
        else:
            scatter_kwargs["color"] = self.template.get_colors(1)[0]

        # Create scatter plot
        scatter = ax.scatter(data[x], data[y], **scatter_kwargs)

        # Add colorbar if color mapping is used
        if color is not None:
            plt.colorbar(scatter, ax=ax, label=color)

        # Set labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        # Adjust x-axis label rotation if needed
        self._adjust_xlabel_rotation(ax, data[x])

        # Set y-axis limits intelligently
        self._set_ylim_intelligently(ax, data, [y], "auto", None)

        plt.tight_layout()
        return fig, ax


class PiePlot(PlotGenerator):
    """Generate pie charts for academic papers."""

    def create(
        self,
        data: DataDict,
        labels: Optional[str] = None,
        values: Optional[str] = None,
        title: Optional[str] = None,
        autopct: str = "%1.1f%%",
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a pie chart.

        Parameters:
        -----------
        data : DataDict
            Data to plot with 'labels' and 'values' column names
        labels : str
            Column name for labels
        values : str
            Column name for values
        title : str, optional
            Plot title
        autopct : str
            Format for percentage labels
        **kwargs
            Additional matplotlib arguments

        Returns:
        --------
        Tuple[Figure, Axes]
            (figure, axes) objects
        """
        fig, ax = self._prepare_plot("pie")

        # Prepare data - both labels and values must be specified for DataDict
        if labels is None or values is None:
            raise ValueError(
                "Both 'labels' and 'values' must be specified for DataDict input"
            )

        plot_labels = [str(lbl) for lbl in data[labels]]
        plot_values = [val for val in data[values]]

        # Get colors
        colors = self.template.get_colors(len(plot_labels))

        # Create pie chart
        ax.pie(
            plot_values, labels=plot_labels, colors=colors, autopct=autopct, **kwargs
        )

        # Set title
        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig, ax


class HeatmapPlot(PlotGenerator):
    def create(
        self,
        data: DataDict,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cmap: str = "viridis",
        annot: bool = False,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a heatmap.

        Parameters:
        -----------
        data : DataDict or array
            Data to plot as heatmap
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        cmap : str
            Colormap to use
        annot : bool
            Whether to annotate cells with values
        **kwargs
            Additional seaborn arguments

        Returns:
        --------
        Tuple[Figure, Axes]
            (figure, axes) objects
        """
        fig, ax = self._prepare_plot("heatmap")

        plot_data = pd.DataFrame(data)

        sns.heatmap(
            plot_data, ax=ax, cmap=cmap, annot=annot, **kwargs
        )  # Set labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig, ax


class RadarPlot(PlotGenerator):
    """Generate radar/spider plots for academic papers."""

    def create(
        self,
        data: DataDict,
        categories: str,
        values: Union[str, list[str]],
        title: Optional[str] = None,
        legend: bool = True,
        legend_preference: Optional[str] = None,
        legend_outside: bool = False,
        legend_style: Union[str, LegendStyle] = LegendStyle.CLEAN,
        fill_alpha: float = 0.1,
        line_width: float = 2.0,
        marker_size: float = 6.0,
        ylim: Optional[tuple[float, float]] = None,
        grid: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """
        Create a radar/spider plot.

        Parameters:
        -----------
        data : DataDict
            Data to plot
        categories : str
            Column name for categories (radar chart axes)
        values : str or list of str
            Column name(s) for values. If list, creates multiple radar lines
        title : str, optional
            Plot title
        legend : bool
            Whether to show legend (only for multiple value series)
        legend_preference : str, optional
            Specific legend location preference
        legend_outside : bool
            Whether to place legend outside plot area
        legend_style : str or LegendStyle
            Style of the legend
        fill_alpha : float
            Transparency for filled areas (default: 0.1)
        line_width : float
            Width of radar lines (default: 2.0)
        marker_size : float
            Size of markers on radar lines (default: 6.0)
        ylim : tuple, optional
            Y-axis limits (min, max) for radar scales
        grid : bool
            Whether to show grid lines
        **kwargs
            Additional matplotlib arguments

        Returns:
        --------
        Tuple[Figure, Axes]
            (figure, axes) objects
        """
        # Get categories and values
        category_labels = data[categories]
        value_cols = [values] if isinstance(values, str) else values
        n_categories = len(category_labels)
        n_series = len(value_cols)

        # Calculate angles for each category
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        # Close the circle
        angles += angles[:1]

        # Create polar subplot
        fig, ax = plt.subplots(
            figsize=self.template.dimensions, subplot_kw=dict(projection="polar")
        )

        # Apply template styling
        self.template.apply_style("radar")

        # Get colors for each series
        colors = self.template.get_colors(n_series)

        # Plot each value series
        for i, col in enumerate(value_cols):
            values_list = [float(v) for v in data[col]]
            # Close the circle
            values_list += values_list[:1]

            # Plot line
            ax.plot(
                angles,
                values_list,
                "o-",
                linewidth=line_width,
                markersize=marker_size,
                label=col if legend and n_series > 1 else None,
                color=colors[i],
                **kwargs,
            )

            # Fill area
            ax.fill(angles, values_list, alpha=fill_alpha, color=colors[i])

        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([str(label) for label in category_labels])

        # Apply font sizes to radar chart labels using new configuration
        radar_category_size = self.template.font_sizes.get(
            "radar_category", self.template.font_sizes["tick"]
        )
        radar_value_size = self.template.font_sizes.get(
            "radar_value", self.template.font_sizes["tick"]
        )

        for label in ax.get_xticklabels():
            label.set_fontsize(radar_category_size)

        for label in ax.get_yticklabels():
            label.set_fontsize(radar_value_size)

        # Set y-axis limits if specified
        if ylim:
            ax.set_ylim(*ylim)

        # Configure grid
        if grid:
            ax.grid(True)
        else:
            ax.grid(False)

        # Set title
        if title:
            ax.set_title(title, pad=20)

        # Add legend if requested and multiple series
        if legend and n_series > 1:
            self._place_legend_intelligently(
                ax,
                legend_preference=legend_preference,
                legend_outside=legend_outside,
                legend_style=legend_style,
            )

        plt.tight_layout()
        return fig, ax
