from typing import Any, Union

from collections.abc import Callable
from itertools import compress
from math import ceil, floor
from os import getcwd, listdir, mkdir, rename, scandir
from os.path import isdir
from re import search, sub
from tqdm import tqdm
from numpy import isnan as npisnan
from numpy import vectorize as npvectorize

import pandas as pd
from plotly.graph_objects import Layout, Scatter, Table, Figure
from plotly.subplots import make_subplots


class Visualisation:
    """
    Class storing all plotting parameters (i.e. layout options) for use when \
    plotting the data.

    Ensure plotly version >=5.12.0 (although tested on 5.15.0)
    """

    def __init__(
        self,
        label_map = None,
        colours: Union[list[str], None] = None,
        sf: float = 1,
        dict_lay_changes: Union[dict[str, Union[str, float, int, bool]], None] = None,
        to_json: bool = False,
    ) -> None:

        self.label_map = label_map
        self.sf: float = sf
        self.ylims_def = lambda x: [0, x.max() + x.max() / 4]
        self.ignore = []
        self.legends = {}
        self.sepPlots = dict()
        self.colours = colours
        self.to_json = to_json

        # Plot Formatting
        self.dict_lay: dict[str, Any] = dict(
            dpi=226.98, #determining page size relative to pixels
            per_py=100_000,
            overideColours=False,

            #font
            font_family="SF Pro Text",
            font_color="black",

            #title
            title_font_size=25 * self.sf,
            title_xanchor="left",

            #axes
            axes_title_font_size=15 * self.sf,
            axes_showgrid=True,
            axes_gridcolour="rgba(156, 156, 156, 0.5)",
            axes_showline=True,
            axes_showspikes=True,
            axes_gridcolor="rgba(156, 156, 156, 0.5)",
            axes_linecolor="grey",
            axes_gridwidth=0.5*self.sf,
            axes_showticklabels=True,
            axes_tickfont_size=12 * self.sf,
            x_axis_tickangle=0,
            y_axis_tickangle=0,
            x_axis_dtick="M12",
            x_axis_tick0=None,
            x_axis_autorange=True,
            y_axis_autorange=False,
            nticks=5,
            x_axis_ticklabelstep=5,
            y_axis_ticklabelstep=10,
            axes_title_standoff=0.3,

            #background
            plot_bgcolor="white",
            background_fillcolor="#C4ECFF",
            background_opacity=0,
            background_y0=-0.1,
            background_x0=-0.1,
            background_y1=1.012,
            background_x1=1,

            #plot lines and points
            dash_types=["solid", "dash"],
            line_width=3 * self.sf,
            marker_size=8 * self.sf,
            highlightMarker_line_width=5,
            highlightMarker_line_colour="red",
            marker_opacity=0.8,
            error_y_type="data",
            error_y_symmetric=False,
            error_y_thickness=1.5 * self.sf,
            error_y_width=0.5 * self.sf,

            #legend
            legend_font_size=15,
            legend_yanchor="top",
            legend_xanchor="right",
            legend_y=0,
            legend_x=0,
            legend_title_text="Subgroups: ",
            legend_valign="middle",

            #AutoAnalysis output format
            autosize=False,
            height_perPlot = 300*self.sf,
            width=700*self.sf,
            title_xref="paper",
            title_x=0,
            table_font_size=12*self.sf,
            table_columnwidth=[1,3,3],
            table_height=45*self.sf,
            subplot_vertical_spacing=None,
            )

        self.dict_font: dict[str, object] = dict(
            family=self.dict_lay["font_family"], color=self.dict_lay["font_color"]
        )
        self.dict_title: dict[str, object] = dict(
            font_size=self.dict_lay["title_font_size"], xanchor=self.dict_lay["title_xanchor"]
        )

    def highlight_point(self, name_col, highlight, col):
        dict_marker = self.make_dict_marker(color=col[name_col])
        if highlight is not None:
            if isinstance(name_col, str):
                name_col_ = name_col.split(",")
            else:
                name_col_ = name_col
            highlight_point = all([any(''.join(e for e in substring if
                                           e.isalnum()) in
                                   ["".join(e for e in x if e.isalnum()) for x in name_col_]
                                   for substring in filter_) for
                               filter_ in highlight])

            if highlight_point and len(highlight) > 0:
                dict_marker["line"] = {"color": self.dict_lay["highlightMarker_line_colour"],
                                       "width": self.dict_lay["highlightMarker_line_width"],}
            else:
                dict_marker["line"] = None

        return dict_marker



    def getColours(self, n=None):
        """
        Generate list of (virtually) infinite colours
        Generates colours only if self.colours has not been defined (is None)

        Parameters:
            n (int): Number of colours to generate
        Returns:
            (list): Hex colours of length >n (where length%6 = 0)
        """
        maxi = 235
        mini = 100
        z = 50

        if (self.colours is None or n>len(self.colours)):
            if n<=10 and not self.dict_lay["overideColours"]:
                cols = [
                        "#5778a4",
                        "#e49444",
                        "#d1615d",
                        "#85b6b2",
                        "#6a9f58",
                        "#e7ca60",
                        "#a87c9f",
                        "#f1a2a9",
                        "#967662",
                        "#b8b0ac",
                        ]
            else:
                def appendHex(x, y, z):
                    x=hex(int(round(x)))[2:]
                    y=hex(int(round(y)))[2:]
                    if len(x) == 1:
                        x = f"{x}0"
                    if len(y) == 1:
                        y = f"{y}0"

                    z = hex(int(round(z)))[2:]
                    if len(z) == 1:
                        z = f"{z}0"
                    out = [
                        f"#{x}{y}{z}",
                        f"#{z}{x}{y}",
                        f"#{y}{z}{x}",
                    ]
                    return out

                cols = []

                #Get initial cols for 3 groups (rgb)
                x=maxi
                y=mini
                cols = cols + appendHex(x, y, z)
                x=maxi
                y=maxi
                cols = cols + appendHex(x, y, z)

                i = 6
                while i <= n:
                    a = x - ((x-mini)/2)
                    b = y
                    cols = cols + appendHex(a, b, z)
                    cols = cols + appendHex(b, a, z)

                    x = x - ((x-mini)/2)
                    y = y - ((y-mini)/2)
                    i+=6

        else:
            cols = self.colours

        return cols

    def check_cname(self, data: pd.DataFrame, c_name="OVERALL"):
        """
        Generate uni-value column for input data
        Used where input data is not stratified
        Where a "stratified-by" column does not exist

        Parameters:
            data (pd.DataFrame): Input Analogy data
            c_name (str): Value for new column (defaults to "OVERALL")
        Returns:
            pd.DataFrame: Data with new column

        Notes:
            Used to make "OVERALL" inputs compatible with stratified inputs
        """
        if c_name not in data.columns:
            #print(f"Warning: {c_name} column does not exist. Making temp.")
            data[c_name] = [c_name] * len(data.index)

        return data

    def rm_NumNull(self, data: pd.DataFrame, metric_type: Union[str, None] = None) -> pd.DataFrame:
        """
        Remove any 0 or NA values in Numerator

        Parameters:
            data (pd.DataFrame): Input data
            metric_type (str): Name of column to plot on x-axis

        Returns:
            pd.DataFrame: Processed Data
        """
        if metric_type is not None:
            df_temp = data.loc[(data[metric_type].notna() & \
                    data[metric_type].notnull())]
        else:
            df_temp = data
        #numerator = [x if x != 0 else pd.NA for x in df_temp["Numerator"]]
        #df_temp_map = pd.notna(numerator)
        #df_temp = df_temp.loc[df_temp_map]

        return df_temp

    def updateLayout_subplots(self, plots, layout, out_row_n, subcat, shapes):
        """
        Update layout of subplots using output from plot_incprev()

        Parameters:
            plots (plotly.Figure): Plotly subplots
            layout_out (tuple): 1st element plotly.Layout
            out_row_n (int): Number of rows of subplots Figure
            subcat (str): Name of stratification factor (used for title)
            shapes (list): List of shapes to add to subplots
        Returns:
            plotly.Figure: Subplots with updated layout

        Notes:
            plot_incprev() returns a tuple, where the 1st element contains the\
            layout options. If using without plot_incprev(), pass in \
            layout_out as a tuple of 1 element (layout_out).
        """
        if subcat is not None:
            title_text = subcat
        else:
            title_text = None

        #Prevents overwriting x and y axes of all subplots
        layout.xaxis=None
        layout.yaxis=None

        plots.update_layout(
            layout,
            autosize=self.dict_lay["autosize"],
            height=self.dict_lay["height_perPlot"] * out_row_n,
            width=self.dict_lay["width"],
            title_text=title_text,
            title_font_family=self.dict_lay["font_family"],
            title_font_size=self.dict_lay["title_font_size"],
            title_xref=self.dict_lay["title_xref"],
            title_xanchor=self.dict_lay["title_xanchor"],
            title_x=self.dict_lay["title_x"],
            shapes=shapes,
            plot_bgcolor=self.dict_lay["plot_bgcolor"],
        )
        return plots

    def make_dict_dash_type(self, data: pd.DataFrame) -> dict[str, str]:
        """
        Make layout dictionary

        Parameters:
            data (pd.DataFrame): Input data

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_dash_type()
        """
        lay: list[str] = self.dict_lay["dash_types"]
        dash_type = dict(zip(set(data["results"]), lay))
        return dash_type

    def make_dict_col(self, data: pd.DataFrame, c_name: str) -> Union[dict[str, str], None]:
        """
        Make layout dictionary

        Parameters:
            data (pd.DataFrame): Input data
            c_name (str): c_name

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_col()
        """
        colours = self.getColours(n=len(set(data[c_name])))
        col = dict(zip(set(data[c_name]), colours))
        return col

    def make_dict_line(self, color: str, dash: str) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            color (str): color
            dash (str): dash

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_line()
        """
        opacity = self.dict_lay["marker_opacity"]
        colour = color[1:]
        colour = [str(int(colour[0:2], 16)),
            str(int(colour[2:4], 16)),
            str(int(colour[4:6], 16)),
            str(opacity)]
        colour = ", ".join(colour)


        dict_line = dict(
                dash=dash,
                color=f"rgba({colour})",
                width=self.dict_lay["line_width"])
        return dict_line

    def make_dict_marker(self, color: str) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            color (str): color

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_marker()
        """
        dict_marker = dict(
            color=color,
            size=self.dict_lay["marker_size"],
            opacity=self.dict_lay["marker_opacity"],
        )
        return dict_marker

    def make_dict_error_y(
        self, array: pd.Series, arrayminus: pd.Series
    ) -> dict[str, Union[str, bool, pd.Series, float, int]]:
        """
        Make layout dictionary for ...

        Parameters:
            array (pd.Series): array
            arrayminus (pd.Series): arrayminus

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_error_y()
        """
        dict_error_y = dict(
            type=self.dict_lay["error_y_type"],
            symmetric=self.dict_lay["error_y_symmetric"],
            array=array,
            arrayminus=arrayminus,
            thickness=self.dict_lay["error_y_thickness"],
            width=self.dict_lay["error_y_width"],
        )
        return dict_error_y

    def make_dict_axes(
        self,
        ylims: list[Union[float, int]],
        x_start=None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """
        Make layout dictionary for

        Parameters:
            ylims (list): ylims

        Returns:
            dict: dictionary of layout options
            dict: dictionary of layout options

        Usage:
            make_dict_axes()
        """
        if x_start is not None:
            x_tick0 = x_start
        else:
            x_tick0 = self.dict_lay["x_axis_tick0"]

        dict_xaxis = dict(
            title_font_size=self.dict_lay["axes_title_font_size"],
            title_font_family=self.dict_lay["font_family"],
            title_standoff=self.dict_lay["axes_title_standoff"],
            autorange=self.dict_lay["x_axis_autorange"],
            showgrid=self.dict_lay["axes_showgrid"],
            showline=self.dict_lay["axes_showline"],
            showspikes=self.dict_lay["axes_showspikes"],
            gridcolor=self.dict_lay["axes_gridcolor"],
            linecolor=self.dict_lay["axes_linecolor"],
            gridwidth=self.dict_lay["axes_gridwidth"],
            showticklabels=self.dict_lay["axes_showticklabels"],
            tickfont_size=self.dict_lay["axes_tickfont_size"],
            tickfont_family=self.dict_lay["font_family"],
            tickangle=self.dict_lay["x_axis_tickangle"],
            dtick=self.dict_lay["x_axis_dtick"],
        )
        dict_yaxis = dict_xaxis.copy()
        dict_yaxis["tickangle"] = self.dict_lay["y_axis_tickangle"]
        dict_yaxis["autorange"] = self.dict_lay["y_axis_autorange"]
        def y_round(x, nticks=10):
            n=0
            if x != 0:
                while x>=10:
                    x = x/10
                    n+=1
                while x<1:
                    x = x*10
                    n-=1

            maxi = 0.5*ceil(x/0.5)
            x = maxi*(10**n) / nticks
            maxi = maxi*(10**n)

            return [x, maxi]
        ticks = y_round(ylims[1]-ylims[0],
                        nticks=self.dict_lay["nticks"])
        subdivide_y = self.dict_lay["y_axis_ticklabelstep"]

        dict_yaxis["dtick"] = ticks[0] / subdivide_y
        dict_yaxis["range"] = [ylims[0], ticks[1]]
        dict_yaxis["ticklabelstep"] = subdivide_y

        dict_xaxis["tick0"]=x_tick0
        dict_xaxis["ticklabelstep"]=self.dict_lay["x_axis_ticklabelstep"]

        return (dict_xaxis, dict_yaxis)

    def make_dict_background(self, i1: Union[int, str], i2: Union[int, str] = 1) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            i1 (int): i1
            i2 (int): i2

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_background()
        """
        if i1 == 1:
            i1 = ""
        if i2 == 1:
            i2 = ""

        dict_fill = dict(
            type="rect",
            yref=f"y{i1} domain",
            xref=f"x{i2} domain",
            y0=self.dict_lay["background_y0"],
            x0=self.dict_lay["background_x0"],
            y1=self.dict_lay["background_y1"],
            x1=self.dict_lay["background_x1"],
            fillcolor=self.dict_lay["background_fillcolor"],
            opacity=self.dict_lay["background_opacity"],
            layer="below",
            line_width=0,
        )
        return dict_fill

    def make_dict_table(self):
        """
        Make layout dictionary for table

        Returns:
            dict: dictionary of layout options

        Usages:
            make_dict_background()
        """
        font = dict(
            family=self.dict_lay["font_family"],
            size=self.dict_lay["table_font_size"],
            color=self.dict_lay["font_color"],
        )

        dict_table = dict(
            header = dict(
                font = font.copy(),
                height=self.dict_lay["table_height"]*1.2,
            ),
            cells = dict(
                font = font.copy(),
                height=self.dict_lay["table_height"],
            ),
            columnwidth=self.dict_lay["table_columnwidth"],
        )
        dict_table["header"]["font"]["size"] = font["size"]*1.2
        return dict_table

    def plot_incprev(
        self,
        data: pd.DataFrame,
        studyName: str,
        subgroups: bool = False,
        c_name: str = "OVERALL",
        ylims: Union[list[float], None] = None,
        incprev: bool = True,
        legend: bool = True,
        withLine: bool = True,
        const = 100_000,
        yearFilter_low = 2006,
        yearFilter_high = 2020,
        highlight = None,
    ) -> tuple[list[Scatter], Layout,]:
        """
        Prepare input inc/prev data
        Uses plotly library to generate traces and layout for inc/prev data.
        Can plot, on y-axis, overall inc/prev, subgrouped inc/prev, or a user-defined numerical column (e.g. Numerator)
        plotted against time (x-axis).

        Parameters:
            data (pd.DataFrame): Data to be visualised
            studyName (str): One of c("Incidence", "Prevalence", "Numerator", "Denominator", "PersonYears")
            subgroups (bool): is data subgrouped?
            c_name (str): column name of the subgroup levels
            ylims (list): vector of c(y_min, y_max), designating y-axis range.
                        If null, then will automatically set c(0, max(data[`studyName`])+(max(data[`studyName`])/5) )
            incprev (bool): Are incidence/prevelance values being plotted on the y-axis?
            legend (bool): Should traces include legends? (default True)
            withLine (bool): Should plot include line graph elements?

        Returns:
            plot (list): list of plotly.trace
            layout (plotly.Layout): Defines layout of traces
            dict_xaxis (dict): Defines plotly x-axis formatting
            dict_yaxis (dict): Defines plotly y-axis formatting

        Usage:
            plot_incprev(data, studyName) OR plot_incprev(data, studyName, TRUE, c_name)

        Details:
            If subgroups==TRUE, must define c_name.
            If plotting numerator/denominator, then set incprev==FALSE.
            If incprev==TRUE, studyName must be one of c("Incidence", "Prevalence")

        Notes:
            If comparing datasets with same levels and format, the function can accomodate 2 of these,
            where both will be plot on same axes but with different linetypes and point shapes.
            A field 'results' should be present defining the dataset data belongs too.
            If data input does not have a 'results' column, function will treat all data as coming from the same dataset.
        """

        if studyName == "Incidence":
            ylab_text = f"Incidence / {const} Person Years"
        elif studyName == "Prevalence":
            ylab_text = f"Prevalence / {const} Population"
        else:
            ylab_text = studyName

        #data_map = [True if x is not pd.NA or not npisna(x) else False for x in data[studyName]]
        data = data[(data[studyName].notna() | ~data[studyName].apply(npisnan))]
        if yearFilter_low is not None:
            data = data[(data["Year"].dt.year>=yearFilter_low)]
        if yearFilter_high is not None:
            data = data[(data["Year"].dt.year<=yearFilter_high)]
        #data = data.loc[data_map]

        if ylims is None:
            ylims = self.ylims_def(data[studyName])

        upper_limit = [x if x > ylims[1] else pd.NA for x in data[studyName]]
        data = data.assign(upper_limit=upper_limit)

        if "results" not in data.columns:
            data = data.assign(results=["NotRequired"] * len(data.index))

        dash_type = self.make_dict_dash_type(data)
        #sort data by c_name to ensure legends in correct order
        try:
            data[c_name] = pd.to_numeric(data[c_name])
        except:
            pass
        data.sort_values(by=c_name, inplace=True)

        if subgroups == True and c_name in self.legends.keys()\
                and self.legends[c_name] is not None:
            col = self.legends[c_name]
            #check all levels are in dict
            #if not, add levels to c_name's colour dictionary
            levels = []
            for name_res, group_res in data.groupby("results"):
                for name_col, group_col in group_res.groupby(c_name):
                    levels.append(name_col)
                break
            stored = set(col.keys())
            levels = set(levels)
            check = levels == col
            if check == False:
                diff = levels.difference(col)
                for i, x in enumerate(diff):
                    col[x] = self.getColours(len(stored)+i)[len(stored)+i]
                self.legends[c_name] = col

        elif subgroups == True and c_name not in self.legends.keys():
            col = self.make_dict_col(data, c_name)
            self.legends[c_name] = col
        else:
            col = self.make_dict_col(data, c_name)

        if col is None:
            print(f"Failed for {c_name}")
            return None, None

        # Plotting
        if subgroups == False:
            plot = []
            for name, group in data.groupby("results"):
                group.sort_values(by="Year", inplace=True)

                dict_line = self.make_dict_line(color=self.getColours(n=1)[0], dash=dash_type[name])

                dict_error_y = self.make_dict_error_y(
                    array=group["Upper"] - group[studyName],
                    arrayminus=group[studyName] - group["Lower"],
                )
                dict_error_y["color"] = dict_line["color"]

                dict_marker = self.make_dict_marker(color=self.getColours(n=1)[0])

                plot.append(
                    Scatter(
                        x=group["Year"],
                        y=group[studyName],
                        mode="markers",
                        marker=dict_marker,
                        error_y=dict_error_y,
                        showlegend=False,
                    )
                )
                if withLine:
                    plot.append(
                        Scatter(
                            x=group["Year"],
                            y=group[studyName],
                            mode="lines",
                            marker=dict_marker,
                            line=dict_line,
                            showlegend=legend,
                            legendgroup="group",
                            name="overall",
                        )
                    )

        elif incprev == True:
            plot = []
            for name_res, group_res in data.groupby("results"):
                for name_col, group_col in group_res.groupby(c_name):
                    skip = False
                    if name_col not in self.ignore:
                        for x in self.ignore:
                            if str(name_col).find(f"'{x}'") != -1:
                                skip=True
                                break
                    else:
                        skip=True
                    if skip==True:
                        continue

                    group_col.sort_values(by="Year", inplace=True)

                    dict_line = self.make_dict_line(color=col[name_col], dash=dash_type[name_res])

                    dict_error_y = self.make_dict_error_y(
                        array=group_col["Upper"] - group_col[studyName],
                        arrayminus=group_col[studyName] - group_col["Lower"],
                    )
                    dict_error_y["color"] = dict_line["color"]

                    dict_marker = self.highlight_point(
                            name_col,
                            highlight,
                            col
                    )
                    plot.append(
                        Scatter(
                            x=group_col["Year"],
                            y=group_col[studyName],
                            mode="markers",
                            marker=dict_marker,
                            error_y=dict_error_y,
                            showlegend=False,
                        )
                    )
                    if withLine:
                        plot.append(
                            Scatter(
                                x=group_col["Year"],
                                y=group_col[studyName],
                                mode="lines",
                                marker=dict_marker,
                                line=dict_line,
                                showlegend=legend,
                                legendgroup="group",
                                name=name_col,
                            )
                        )

            plot = []
            for name_res, group_res in data.groupby("results"):
                for name_col, group_col in group_res.groupby(c_name):
                    skip = False
                    if name_col not in self.ignore:
                        for x in self.ignore:
                            if str(name_col).find(f"'{x}'") != -1:
                                skip=True
                                break
                    else:
                        skip=True
                    if skip==True:
                        continue

                    group_col.sort_values(by="Year", inplace=True)

                    dict_line = self.make_dict_line(color=col[name_col], dash=dash_type[name_res])
                    dict_marker = self.highlight_point(
                        name_col,
                        highlight,
                        col
                    )

                    plot.append(
                        Scatter(
                            x=group_col["Year"],
                            y=group_col[studyName],
                            mode="markers",
                            marker=dict_marker,
                            showlegend=False,
                        )
                    )
                    if withLine:
                        plot.append(
                            Scatter(
                                x=group_col["Year"],
                                y=group_col[studyName],
                                mode="lines",
                                marker=dict_marker,
                                line=dict_line,
                                showlegend=legend,
                                legendgroup="group",
                                name=name_col,
                            )
                        )

        dict_xaxis, dict_yaxis = self.make_dict_axes(ylims,
                                                     data["Year"].min())
        dict_xaxis["title_text"] = "Year"
        dict_yaxis["title_text"] = f"{ylab_text}"
        layout = Layout(
            # title_text=f"{studyName} Trend by {c_name}",
            legend_title_text="Subcategories:",
            legend_title_font_size=self.dict_lay["legend_font_size"] *1.3 * self.sf,
            legend_font_size=self.dict_lay["legend_font_size"] * self.sf,
            legend_font_family=self.dict_lay["font_family"],
            legend_valign=self.dict_lay["legend_valign"],
            scattermode="group",
            scattergap=0.75,
            font=self.dict_font,
            title=self.dict_title,
            xaxis=dict_xaxis,
            yaxis=dict_yaxis,
        )

        return (plot, layout)

    def plot_scatter(
        self,
        data: pd.DataFrame,
        y_var: str,
        x_var: str,
        c_name: str,
        is_numeric_x: bool = True,
        ylims: Union[list[float], None] = None,
        xlims: Union[list[float], None] = None,
        legend: bool = True,
        dir: str = ".",
        interactive = False,
        meta_vars = None,
        returnPlot = False,
        returnTrace = False,
        highlight = None
    ) -> tuple[list[Scatter], Layout, dict[str, object], dict[str, object],]:
        """
        Plot a simple scatter graph
        Non-automated implementation of plot_incprev()

        Parameters:
            data (pd.DataFrame):
            y_var (str):
            x_var (str):
            c_name (str):
            is_numeric_x (bool):
            ylims (list):
            xlims (list):
            legend (bool):
            dir (str): path to dir to save graphs to disk (if applicable)
            interactive (bool): Output interactive? If returnTrace==False
            meta_vars (list): List of column names to show in hover labels\
                if interactive output
            returnPlot (bool): Return Figure object? If returnTrace==False
            returnTrace (bool): Return list of traces or save figure to disk

        Returns:
            list: List of traces
                if returnTrace is True

        Notes:
            returnTrace determines whether a list of traces is returned, or\
            the final figure. The final figure can be returned as a plotly \
            Figure object (returnPlot is True), saved to disk as .html \
            (interactive is True AND returnPlot is False) or saved to disk as \
            .png (interactive is True AND returnPlot is False)
        """
        ylab_text = y_var

        data_map = [True if x is not pd.NA else False for x in data[y_var]]
        data = data.loc[data_map]

        if ylims is None:
            ylims = self.ylims_def(data[y_var])
        if is_numeric_x:
            if xlims is None:
                xlims = self.ylims_def(data[x_var])

        upper_limit = [x if x > ylims[1] else pd.NA for x in data[y_var]]
        data = data.assign(upper_limit=upper_limit)

        if "results" not in data.columns:
            data = data.assign(results=["NotRequired"] * len(data.index))

        dash_type = self.make_dict_dash_type(data)

        try:
            data[c_name] = pd.to_numeric(data[c_name])
        except:
            pass
        data.sort_values(by=c_name, inplace=True)

        if c_name in self.legends.keys() and self.legends[c_name] is not None:
            col = self.legends[c_name]
            #check all levels are in dict
            #if not, add levels to c_name's colour dictionary
            levels = []
            for name_res, group_res in data.groupby("results"):
                for name_col, group_col in group_res.groupby(c_name):
                    levels.append(name_col)
                break
            stored = set(col.keys())
            levels = set(levels)
            check = levels == col
            if check == False:
                diff = levels.difference(col)
                for i, x in enumerate(diff):
                    col[x] = self.getColours(len(stored)+i)[len(stored)+i]
                self.legends[c_name] = col

        elif c_name not in self.legends.keys():
            col = self.make_dict_col(data, c_name)
            self.legends[c_name] = col
        else:
            col = self.make_dict_col(data, c_name)

        if col is None:
            print(f"Failed for {c_name}")
            return None, None

        traces = []
        for name_col, group_col in data.groupby(c_name):
            skip = False
            if name_col not in self.ignore:
                for x in self.ignore:
                    if str(name_col).find(f"'{x}'") != -1:
                        skip=True
                        break
            else:
                skip=True
            if skip==True:
                continue

            group_col.sort_values(by=x_var, inplace=True)

            dict_marker = self.make_dict_marker(color=col[name_col])

            if meta_vars is not None:
                meta_label = group_col[meta_vars].applymap(str)
                meta_label = meta_label.agg(', '.join, axis=1).tolist()
            else:
                meta_label = None

            traces.append(
                Scatter(
                    x=group_col[x_var],
                    y=group_col[y_var],
                    mode="markers",
                    marker=dict_marker,
                    showlegend=legend,
                    legendgroup="group",
                    name=name_col,
                    hoverinfo="text",
                    hovertext=meta_label,
                )
            )


        dict_xaxis, dict_yaxis = self.make_dict_axes(ylims)
        dict_xaxis["title_text"] = x_var
        dict_yaxis["title_text"] = y_var
        title_text=f"{y_var} trend by {x_var}"
        layout = Layout(
            title_text=title_text,
            legend_title_text="Subcategories:",
            legend_title_font_size=15 * self.sf,
            legend_font_size=10 * self.sf,
            legend_valign=self.dict_lay["legend_valign"],
            scattermode="group",
            scattergap=0.75,
            font=self.dict_font,
            title=self.dict_title,
            xaxis=dict_xaxis,
            yaxis=dict_yaxis,
        )
        if returnTrace:
            return traces, layout
        else:
            plot = Figure(
                data = traces,
                layout = layout
            )

            label_out = sub("[[:punct:] ]+", "", title_text)
            if interactive:
                if returnPlot:
                    return plot
                else:
                    plot.write_html(f"{dir}/{label_out}_plots.html")
            else:
                plot.write_image(f"{dir}/{label_out}_plots.pdf", engine="kaleido")

    def table(
            self,
            data: pd.DataFrame,
            columns: list[str],
            ndec: int = 2,
        ) -> Table:
        """
        """
        layout = self.make_dict_table()
        layout["header"]["values"]=columns
        layout["cells"]["values"]=[data[x].tolist() for x in data.columns]

        table = Table(**layout)

        return table

    def table_Incprev(
            self,
            data: pd.DataFrame,
            formatData: bool = True,
            columns: list[str] = ["Year", "Incidence", "Prevalence",],
            i_incidence = None,
            i_prevalence = None,
        ) -> Table:
        """
        Makes Table trace for IncPrev data
        Summary table for describing IncPrev for a single disease across all times\
        in data

        Parameters:
        data (pd.DataFrame): Data to be visualised
        formatData (bool): Format the input data to plot as a table
        columns (list[str]): Columns of table
        ndec (int):

        Returns:
        plot (Table): plotly Table trace

        Notes:
            If formatData==False, make sure columns reflects the column names of \
            input data.
        """
        if i_incidence is None:
            data_inc = data.loc[data["Incidence"].notna(), :]
        else:
            data_inc = data.loc[data.iloc[:, i_incidence].notna(), :]
        if i_prevalence is None:
            data_prev = data.loc[data["Prevalence"].notna(), :]
        else:
            data_prev = data.loc[data.iloc[:, i_prevalence].notna(), :]

        def formatdat(dat: pd.DataFrame,
                datcols: list[str],
                ):
            label = datcols[0]
            i_col = dat.columns.get_indexer(datcols)
            values  = dat.iloc[:, i_col].applymap(lambda x: round(x, 2))
            years = dat.loc[:,"Year"].map(lambda x: x.year)

            dat = pd.concat([
                    years,
                    values.apply(
                            lambda x: f"{x[0]} ({x[1]}, {x[2]})",
                            axis=1)
                    ], axis = 1)
            dat.columns = ["Year", label]

            return dat

        if formatData == True:
            data_inc = formatdat(data_inc, ["Incidence", "Upper", "Lower"])
            data_prev = formatdat(data_prev, ["Prevalence", "Upper", "Lower"])
            data_table = pd.merge(data_inc, data_prev, how="outer")
            data_table = data_table.sort_values("Year", axis=0, ascending=False)
        else:
            data_table = data

        table = self.table(data_table,
                           columns)

        return table

    def sequesteredGroups_plot(
        self,
        data: pd.DataFrame,
        studyName: str,
        groupings: list[list[str]],
        subgroups: bool = False,
        c_name: str = "OVERALL",
        ylims: Union[list[float], None] = None,
        incprev: bool = True,
        metric_type: Union[str, None] = None,
        legend: bool = True,
    ) -> tuple[list[Scatter], list[Union[Layout, dict[str, Union[str, int, float, bool]]]],]:
        """
        Generate a group of plots
        Generate a list of plots, where the levels in each plot are defined by a list of character vectors of levels in c_name.

        Parameters:
            data (pd.DataFrame): Data to be visualised
            studyName (str): One of c("Incidence", "Prevalence", "Numerator", "Denominator", "PersonYears")
            subgroups (bool): is data subgrouped?
            groupings (list): Each element a list of strings defining levels to plot
            c_name (str): column name of the subgroup levels
            ylims (list): vector of c(y_min, y_max), designating y-axis range.
                        If null, then will automatically set c(0, max(data[`studyName`])+(max(data[`studyName`])/5) )
            incprev (bool): Are incidence/prevelance values being plotted on the y-axis?
            metric_type (str): Name of column to plot on x-axis (if not studyName)
            legend (bool): Should traces include legends? (default True)

        Returns:
            plot (list): list of plotly.trace
            layout (Layout):

        Usage:
            sequesteredGroups_plot(data, studyName, groupings)
            sequesteredGroups_plot(data, studyName, groupings, TRUE, c_name=c_name, incPrev=FALSE, metric_type=metric_type)

        Notes:
            If is.null(ylims)==TRUE then function will automatically determine the max y-axis value across the groups; c(0, ymax)

            If subgroups==TRUE, must define c_name.
            Or if subgroups==False, and c_name!="OVERALL", then must define the c_name of input data

            If plotting numerator/denominator, then set incprev==FALSE.
            If incprev==TRUE, studyName must be one of c("Incidence", "Prevalence")
        """
        def ignoreRow(x, labs=self.ignore):
            skip = False
            if x in labs:
                skip = True
            else:
                for lab in labs:
                    if str(x).find(f"'{lab}'") != -1:
                        skip=True
                        break
            return skip


        if metric_type is None:
            metric_type = studyName

        # ensure consistent y-axis for grouped graphs if ylims==None
        if ylims is None:
            y_max = 0
            for i in range(0, len(groupings)):
                df_temp = self.rm_NumNull(data, metric_type=metric_type).copy()

                rem = df_temp[c_name].map(ignoreRow)
                rem = rem[rem].index
                df_temp.drop(labels=rem, axis=0, inplace=True)

                y_temp = self.ylims_def(df_temp[studyName])[1]
                if y_max < y_temp:
                    y_max = y_temp

            ylims = [0, y_max]

        plots = []
        for i in range(0, len(groupings)):
            levels = groupings[i]

            # remove na values in Numerator col
            df_temp = self.rm_NumNull(data, metric_type=metric_type)

            #if numeric c_name col, convert to int if possible, then to str
            #to int removes decimal if being treated as float
            #not pretty or efficient, but works
            try:
                df_temp[c_name] = df_temp[c_name].map(lambda x: int(x))
                df_temp[c_name] = df_temp[c_name].map(lambda x: str(x))
            except:
                df_temp[c_name] = df_temp[c_name].map(lambda x: str(x))


            # remove values not in current levels list
            def subLev(x, levels=None):
                if x in levels:
                    return x
                else:
                    return pd.NA
            subsetLevels = npvectorize(subLev, excluded=['levels'])
            numerator = df_temp[c_name].to_numpy()
            numerator = subsetLevels(numerator, levels=levels)
            #numerator = [x if x in levels else pd.NA for x in df_temp[c_name]]
            df_temp_map = pd.notna(numerator)
            df_temp = df_temp.loc[df_temp_map]
            p, layout = self.plot_incprev(
                df_temp, studyName, subgroups, c_name, ylims, incprev, legend=legend
            )
            if p is None:
                return None, None

            plots.append(p)

        return (plots, layout)

    def autoAnalysis(self, subdirs: bool = False, subdivideLevels: bool = False,
        inMemory=False, file_inc=None, file_prev=None, yearFilter=None) -> None:
        """
        Visualise incPrev data in the current working directory.
        Produces a .pdf file of graphs for each subgroup and incPrev combination.

        Parameters:
            subdirs (bool): Is the data organised into dirs of different conditions?
            subdivideLevels (bool): Should the subgroup levels be split between 2 graphs?

        Usage:
            autoAnalysis()

        Notes:
            Filename structure: {Condition}_{SUBGROUP}_*.csv
        """

        if subdirs == True:
            dirs = [f.path for f in scandir(getcwd()) if f.is_dir() and not f.name.startswith(".")]
        else:
            dirs = ["."]

        if inMemory:
            dsr_inc = pd.read_csv(file_inc, index_col=[0,1,2,3])
            dsr_prev = pd.read_csv(file_prev, index_col=[0,1,2,3])

            subcats = set([index[3] for index in dsr_inc.index])
            dirs = set([index[0] for index in dsr_inc.index])

        for dir in tqdm(dirs):
            if not inMemory:
                files_csv = [x for x in listdir(dir) if x[-4:] == ".csv"]
                subcats = [sub(".*?\\_(.*)(Inc|Prev).*", "\\1", x) for x in files_csv]
                subcats_map = [True if len(x) == 0 else False for x in subcats]
                if True in subcats_map:
                    files_change = list(compress(files_csv, subcats_map))
                    for file in files_change:
                        name_new = sub("(.*?\\_).*(Inc|Prev.*$)", "\\1OVERALL_\\2", file)
                        rename(f"{dir}/{file}", f"{dir}/{name_new}")

                files_csv = [x for x in listdir(dir) if x[-4:] == ".csv"]
                subcats = [sub(".*?\\_(.*)_(Inc|Prev).*", "\\1", x) for x in files_csv]

            for subcat in subcats:
                if subcat in self.sepPlots.keys():
                    n_rows = self.sepPlots[subcat]
                else:
                    n_rows = 2

                out_row_n = 1
                plots = make_subplots(
                    cols=1, rows=n_rows, vertical_spacing=0.105, subplot_titles=["..."] * n_rows
                )
                shapes = []

                if subcat == "OVERALL":
                    query_str = f"{subcat}"

                    if inMemory:
                        dat_subcat = self.combine(query_str=query_str,
                            inMemory=inMemory,
                            cond=dir,
                            file_prev=dsr_prev,
                            file_inc=dsr_inc)
                    else:
                        dat_subcat = self.combine(dir, query_str, inMemory, file_prev, file_inc)
                    #Check both incidence and prevalence data is present
                    if "Incidence" not in dat_subcat.columns or "Prevalence" not in dat_subcat.columns:
                        continue

                    if len([x for x in dat_subcat.columns if x == "OVERALL"]) == 0:
                        dat_subcat = dat_subcat.assign(OVERALL=["overall"] * len(dat_subcat.index))

                    # remove na values in metric col
                    dat_subcat_map = [
                        True if pd.notna(x) else False for x in dat_subcat["Prevalence"]
                    ]
                    dat_temp = dat_subcat.loc[dat_subcat_map]
                    p, layout = self.plot_incprev(dat_temp,
                                                  "Prevalence",
                                                  legend=True,
                                                  )
                    for trace in p:
                        plots.add_trace(trace, row=out_row_n, col=1)
                        plots.layout.annotations[out_row_n - 1].update(text="Prevalence")
                    out_row_n += 1
                    plots.update_xaxes(layout.xaxis, col=1, row=out_row_n - 1)
                    plots.update_yaxes(layout.yaxis, col=1, row=out_row_n - 1)
                    shapes.append(self.make_dict_background(out_row_n - 1))

                    dat_subcat_map = [
                        True if pd.notna(x) else False for x in dat_subcat["Incidence"]
                    ]
                    dat_temp = dat_subcat.loc[dat_subcat_map]
                    p, layout = self.plot_incprev(dat_temp, "Incidence", legend=False)
                    for trace in p:
                        plots.add_trace(trace, row=out_row_n, col=1)
                        plots.layout.annotations[out_row_n - 1].update(text="Incidence")
                    out_row_n += 1
                    plots.update_xaxes(layout.xaxis, col=1, row=out_row_n - 1)
                    plots.update_yaxes(layout.yaxis, col=1, row=out_row_n - 1)
                    shapes.append(self.make_dict_background(out_row_n - 1))

                else:
                    query_str = f"{subcat}"
                    if inMemory:
                        dat_subcat = self.combine(query_str=query_str,
                            inMemory=inMemory,
                            cond=dir,
                            dat_prev=dsr_prev,
                            dat_inc=dsr_inc)
                    else:
                        dat_subcat = self.combine(dir, query_str, inMemory, file_prev, file_inc)
                    #Check both incidence and prevalence data is present
                    if "Incidence" not in dat_subcat.columns or "Prevalence" not in dat_subcat.columns:
                        continue
                    levels_dat = dat_subcat[subcat]

                    if subcat in self.sepPlots.keys():
                        groupings = self.generateGroups(levels_dat, ceil(len(levels_dat) / self.sepPlots[subcat]))
                    else:
                        groupings = self.generateGroups(levels_dat, len(levels_dat))

                    out, layout = self.sequesteredGroups_plot(
                        dat_subcat,
                        "Prevalence",
                        groupings,
                        subgroups=True,
                        c_name=subcat,
                        legend=True,
                    )
                    if out is None:
                        continue

                    for group in out:
                        for trace in group:
                            plots.add_trace(trace, row=out_row_n, col=1)
                            plots.layout.annotations[out_row_n - 1].update(text="Prevalence")
                        out_row_n += 1
                    plots.update_xaxes(layout.xaxis, col=1, row=out_row_n - len(out))
                    plots.update_yaxes(layout.yaxis, col=1, row=out_row_n - len(out))
                    shapes.append(self.make_dict_background(out_row_n - 1))

                    out, layout = self.sequesteredGroups_plot(
                        dat_subcat,
                        "Incidence",
                        groupings,
                        subgroups=True,
                        c_name=subcat,
                        legend=False,
                    )
                    if out is None:
                        continue

                    for group in out:
                        for trace in group:
                            plots.add_trace(trace, row=out_row_n, col=1)
                            plots.layout.annotations[out_row_n - 1].update(text="Incidence")
                        out_row_n += 1
                    plots.update_xaxes(layout.xaxis, col=1, row=out_row_n - len(out))
                    plots.update_yaxes(layout.yaxis, col=1, row=out_row_n - len(out))
                    shapes.append(self.make_dict_background(out_row_n - 1))

                plots = self.updateLayout_subplots(plots, layout, out_row_n,
                        subcat, shapes)

                label_out = sub("[[:punct:]]+", "", subcat)
                if inMemory: #For the moment, inMemory indicates DSR data
                    if self.label_map is not None:
                        dir_lab = self.label_map[dir]

                    if not isdir(f"{dir_lab}/DsrPlots"):
                        mkdir(f"{dir_lab}/DsrPlots")

                    if self.to_json:
                        plots.write_json(f"{dir_lab}/DsrPlots/{dir_lab}_{label_out}_DSR_plots.json")
                    else:
                        plots.write_image(f"{dir_lab}/DsrPlots/{dir_lab}_{label_out}_DSR_plots.pdf", engine="kaleido")
                else:
                    if not isdir(f"{dir}/CrudePlots"):
                        mkdir(f"{dir}/CrudePlots")

                    if self.to_json:
                        plots.write_json(f"{dir}/CrudePlots/{label_out}_plots.json")
                    else:
                        plots.write_image(f"{dir}/CrudePlots/{label_out}_plots.pdf", engine="kaleido")

    def summaryAnalysis(self, inMemory=False, file_prev=None, file_inc=None) -> None:
        """

        """
        dirs = [f.path for f in scandir(getcwd()) if f.is_dir() and not f.name.startswith(".")]
        spec_block = ([{}, {"type": "table", "rowspan": 2}], [{}, {}])

        n_rows = len(dirs)*2
        plots = make_subplots(
            cols=2, rows=n_rows, vertical_spacing=self.dict_lay["subplot_vertical_spacing"],
            specs=list(spec_block * int(n_rows/2)),
            subplot_titles=[" "] * n_rows * 2
        )
        shapes = []

        for i, dir in enumerate(dirs):
            dat = self.combine(dir, "OVERALL", inMemory, file_prev, file_inc)

            row = (i*2)+1
            plot, layout = self.plot_incprev(dat, "Incidence", legend=False)
            for trace in plot:
                plots.add_trace(trace, row = row, col=1)
            plots.update_xaxes(layout.xaxis, col=1, row=row)
            plots.update_yaxes(layout.yaxis, col=1, row=row)
            shapes.append(self.make_dict_background(row, 1))

            row = (i*2)+2
            plot, layout = self.plot_incprev(dat, "Prevalence", legend=False)
            for trace in plot:
                plots.add_trace(trace, row=row, col=1)
            plots.update_xaxes(layout.xaxis, col=1, row=row)
            plots.update_yaxes(layout.yaxis, col=1, row=row)
            shapes.append(self.make_dict_background(row, 1))

            table = self.table_Incprev(dat)
            plots.add_trace(table, row=(i*2)+1, col=2)

            text = dir[-dir[::-1].find("/"):]
            plots.layout.annotations[(i*4)].update(text=f"{text}")

        plots = self.updateLayout_subplots(plots, layout, row,
                "OVERALL", shapes)

        label_out = "OverallSummary"
        plots.write_image(f"./{label_out}_plots.pdf", engine="kaleido")


    # Extra Functions
    def combine(
        self,
        subDir: Union[str, None]=None,
        query_str: Union[str, None] = None,
        inMemory=False,
        cond=None,
        dat_prev=None,
        dat_inc=None) -> pd.DataFrame:
        """
        Combine Incidence and Prevalence outputs from auto inc/prev
        Merges datasets defined in grepl_query.
        Query file names.

        Parameters:
            subDir (str): Subdirectory name where .csv files are located
            query_str (str): string to search dir (subgrouping label)

        Returns:
            pd.DataFrame: Merged data.

        Usage:
            combine(subDir,  lambda files: list(
                            itertools.compress(files, [bool(re.search(query_str, x)) for x in files]))

        Notes:
            Filename structure: {Condition}_*.csv
            quert_str must be in full (full str enclosed by \_), but not \
            including \_
        """
        if inMemory:
            if query_str is not None:
                dat_inc = dat_inc.loc[(cond, slice(None), slice(None), query_str), :]
                dat_prev = dat_prev.loc[(cond, slice(None), slice(None), query_str),:]

            dat_inc = dat_inc.assign(results=["Inc"] * len(dat_inc.index))
            dat_prev = dat_prev.assign(results=["Prev"] * len(dat_prev.index))

            df_inter = pd.concat([dat_inc, dat_prev], ignore_index=False)

            df_inter = df_inter.reset_index(level=["Subgroup", "Year"])
            df_inter = df_inter.rename(columns={"Subgroup":query_str,
                    "LowerCI":"Lower",
                    "UpperCI":"Upper"})

        else:
            files_to_read = list({f"{subDir}/{x}" for x in listdir(f"{subDir}/") if x[-4:] == ".csv"})
            if query_str is not None:
                def grepl_query(files, query_str):
                    return list(compress(files, [bool(search(f"_{query_str}(_Inc|_Prev)", x)) for x in files]))

                files_to_read = grepl_query(files_to_read, query_str)

            fileName = sub(".*?\\_(.*)\\_.*", "\\1", files_to_read[0])
            sub_name = f"{subDir}_{fileName}"
            df_inter = pd.read_csv(files_to_read[0])

            df_inter = df_inter.assign(results=[sub_name] * len(df_inter.index))
            start_loop = True

            i = 1
            for file in files_to_read[i : len(files_to_read)]:
                fileName = sub(".*?\\_(.*)\\_.*", "\\1", files_to_read[i])
                sub_name = f"{subDir}_{fileName}"

                df = pd.read_csv(file)

                new_cols = [x for x in df.columns if x not in df_inter.columns]
                for col in new_cols:
                    df_inter = df_inter.assign(**{f"{col}": pd.NA * len(df_inter.index)})

                df_inter = pd.concat([df.assign(results=[sub_name] * len(df.index)), df_inter],
                        ignore_index=True)

                i += 1


        df_inter = self.rm_NumNull(df_inter)
        df_inter["Year"] = pd.to_datetime(df_inter["Year"])

        df_inter = self.check_cname(df_inter, query_str)

        return df_inter

    def generateGroups(self, dat_levels: pd.Series, max_groups: int) -> list[list[str]]:
        """
        Group categorical data into a defined number of groups.
        Where there are many levels to graph, can return levels in groups,
        to distribute data across multiple graphs for clarity.
        Useful to automate input into sequesteredGroups_plot().

        Parameters:
            dat_levels (pd.Series): Subgroup column of incprev data.
            max_groups (int): Max number of levels per graph.

        Returns:
            list: Each element a list of strings consisting of a group of levels.

        Usage:
            generateGroups(dat, max_groups)
        """
        diffVals = dat_levels.unique()
        makeType = None
        def ListType(inp, ignore):
            try:
                out = [int(x) for x in inp if x not in ignore]
                out = sorted(set(out))
                return [str(x) for x in out]
            except:
                pass
            try:
                out = [float(x) for x in inp if x not in ignore]
                out = sorted(set(out))
                return [str(x) for x in out]
            except:
                pass
            out = [x for x in inp if x not in ignore]
            return sorted(set(out))

        diffVals = ListType(diffVals, self.ignore)

        diffVals_n = len(diffVals)
        group_n = ceil(diffVals_n / max_groups)

        group_levels = []
        if group_n > 1:
            for i in range(0, group_n):
                group_levels.append(
                    diffVals[(i * max_groups) - max_groups + 1 : i * max_groups]
                )
        else:
            group_levels.append(diffVals[0:diffVals_n])

        return group_levels

    def set_ylims(
        self,
        subDir: str,
        features: list[str] = ["OVERALL", "SEX", "AGE_CATG", "ETHNICITY", "HEALTH_AUTH"],
    ) -> pd.DataFrame:
        """
        NEEDS UPDATING
        Find ylims for incidence and prevalence data independently
        Get ylims to pass into plotting functions, where
        ylim_max is the same across each feature, but does not cut off any values.
        Does this for both incidence and prevalence output data independently.
        AKA enables plotting of seperate inc/prev graphs with consistent y-axis bounds.

        Parameters:
            subDir (str): Name of the subdirectory that contains the data.
            features (list): Subgrouped datasets (str) to generate consistent y-axis bounds for.

        Returns:
            pd.DataFrame: Dataframe of ylims for each dataset to plot.

        Usage:
            set_ylims(subDir)

        Notes:
            Relies on file name format: Condition_Subgroup_Inc|Prev.csv
        """

        catg_inc = [f"{x}_Inc" for x in features]
        catg_prev = [f"{x}_Prev" for x in features]
        df_y_minMax = pd.DataFrame(
            {"name": [catg_inc, catg_prev], "ylims": [[pd.NA, pd.NA] * len([catg_inc, catg_prev])]}
        )

        files_to_read = list([f"./{x}/" for x in listdir(f"./{dir}/") if x[-4:] == ".csv"])
        func_features = lambda file: sub(".*?\\_(.*)(_Inc|_Prev).*", "\\1", file)
        files_to_read = [x for x in files_to_read if func_features(x) in features]
        for study in ["Inc", "Prev"]:
            if study == "Inc":
                studyName = "Incidence"
            else:
                studyName = "Prevalence"

            y_max = 0

            read_map = [True if search(study, x) is not None else False for x in files_to_read]
            files_to_read_mapped = list(compress(files_to_read, read_map))
            for i in range(0, len(files_to_read_mapped)):
                # get file name and path
                df_temp = pd.read_csv(files_to_read_mapped[i])
                df_temp = self.rm_NumNull(df_temp, metric_type=studyName)

                y_temp = self.ylims_def(df_temp[studyName])[1]
                if y_max < y_temp:
                    y_max = y_temp

                ylims = [0, y_max]

                names = [sub(".*?\\_(.*)\\_.*", "\\1", x) for x in files_to_read_mapped]
                for n in names:
                    df_temp_map = [True if x == n else False for x in df_y_minMax[names]]
                    df_y_minMax["ylims"].loc[df_temp_map] = ylims

        return df_y_minMax


def organise_wdir(cdir: str = "./") -> None:
    """
    Organises incprev results files in current work directory.
    Creates a directory for each condition, and moves incprev outputs into \
    corresponding directories.

    Parameters:
        cdir (str): current directory (the directory to organise).

    Notes:
        Needed to run before autoAnalysis(), otherwise will get incorrect output graphs
    """
    files_to_read = [f"{x}" for x in listdir(cdir) if x[-4:] == ".csv"]
    names = {sub("(.*?)_.*$", "\\1", x) for x in files_to_read}

    getSubdirs = npvectorize(lambda x, cdir_=None: x if isdir(f"{cdir_}{x}") else None,
                excluded=["cdir_"])
    subdirs = getSubdirs(listdir(cdir), cdir)
    subdirs = [x for x in subdirs if x is not None]

    for name in names:
        if name not in subdirs:
            mkdir(f"{cdir}{name}")
        filesMove = [x for x in files_to_read if sub("(.*?)_.*$", "\\1", x) == name]
        for file in filesMove:
            rename(f"{cdir}{file}", f"{cdir}{name}/{file}")


