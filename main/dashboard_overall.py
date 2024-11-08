import sys
import os
import gc
from os import chdir
import polars as pl
import pandas as pd
import plotly as plt
import numpy as np
from math import floor, log10
import json
import yaml
from tableone import TableOne
from pikepdf import Pdf


"""
To do:
    filename: just condition
    sort imd properly in legend
    File structure:
        Condition -
            standardised.csv crude.csv pdfs
    crude.csv -
        Overall, ethnicity, imd, age and sex (filter only for these)
"""

with open("wdir.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)

if not os.path.isdir(f"{config['PATH']}{config['dir_out']}crude"):
    os.mkdir(f"{config['PATH']}{config['dir_out']}crude")
if not os.path.isdir(f"{config['PATH']}{config['dir_out']}dsr"):
    os.mkdir(f"{config['PATH']}{config['dir_out']}dsr")

dataSelect = ["crude", "dsr",]
DATA = dataSelect[0]

PATH = config["PATH"]
DIR_MAIN = f"{PATH}{config['dir_main']}"
DIR_OUT = f"{PATH}{config['dir_out']}{DATA}/"
DIR_DATA = DIR_OUT

if not os.path.isdir(f"{DIR_OUT}Summaries"):
    os.mkdir(f"{DIR_OUT}Summaries")

#sys.path.append(f"{DIR_MAIN}/ANALOGY_SCIENTIFIC/analogy/study_design/incidence_prevalence/")
import AnalogyGraphing as ag

group = "Overall"
title_standoff = 0.4
dpi = 226.98
N_PAGES = 1
PAPER_DIM = {
        "a4": tuple([11.7, 8.3])
        }
per_x = 1_000
inMemory = True
YEAR = 2019
calc_tableone=False #Only need to calculate once, and saves to disk

PAGES_CONFIG = dict(
        _1= dict(
            spec_block=tuple([
                [{}, {"type": "table"}],
                [{}, {"type": "table"}],
                ]),
            plt_titles=tuple([
                f"Point Prevalence", " ",
                "Incidence Rate", " ",
                ]),
            column_widths=tuple([3,1]),
            row_heights=tuple([1,1]),
        ),
        _2= dict(
            spec_block=tuple([
                [{}, {}],
                [{}, {}],
                ]),
            plt_titles=tuple([
                "Point Prevalence - Ethnicity", "Incidence Rate - Ethnicity",
                "Point Prevalence - Deprivation (IMD deciles)", "Incidence Rate - Deprivation (IMD deciles)",
                ]),
            column_widths=tuple([1,1]),
            row_heights=tuple([1,1]),
        ),
)

#####################################################################

def read_dat_dsr(path_dat,
             yearFilter_low=2006,):

    schema = {
            "Condition": pl.Utf8,
            "Subgroup": pl.Utf8,
            "Year": pl.Int64,
            "Group": pl.Utf8,
            "Incidence": pl.Float64,
            "LowerCI_inc": pl.Float64,
            "UpperCI_inc": pl.Float64,
            "LowerCI_prev": pl.Float64,
            "Prevalence": pl.Float64,
            "UpperCI_prev": pl.Float64,
            }

    commonCols = tuple(["Condition",
                        "Subgroup",
                        "Year",
                        "Group",])
    incCols = commonCols + tuple(["Incidence",
                                  "LowerCI_inc",
                                  "UpperCI_inc",])
    prevCols = commonCols + tuple(["Prevalence",
                                  "LowerCI_prev",
                                  "UpperCI_prev",])

    dat = pl.read_csv(path_dat, schema=schema,)

    def formatDat(dat,
                  cols,
                  yearFilter_low,
                  suff):
        if suff=="inc":
            metric = "Incidence"
        elif suff=="prev":
            metric = "Prevalence"

        dat_ = (
                dat
                .select(pl.col(cols))
                .filter(
                    pl.col("Year") >= yearFilter_low
                    )
                .filter(
                    pl.col("Group").is_in([
                        "Deprivation",
                        "Ethnicity",
                        "Overall",
                        ])
                    )
                .filter(
                    pl.col(metric).is_not_null()
                    )
                .rename(
                    {f"UpperCI_{suff}":"UpperCI", f"LowerCI_{suff}":"LowerCI",}
                    )
                .with_columns(
                    pl.col("Year").cast(pl.Utf8)
                    )
                .with_columns(
                    pl.col("Year").str.strptime(pl.Datetime, "%Y")
                    )
                .with_columns(
                    pl.col("Subgroup").str.replace_all("'", "")
                    )
                .filter(
                    pl.col("Subgroup") != "MissingImd"
                    )
                )

        return dat_

    dat_inc = formatDat(dat, incCols, yearFilter_low, "inc")
    dat_prev = formatDat(dat, prevCols, yearFilter_low, "prev")

    return dat_inc, dat_prev


def read_dat_crude(path_dat,
             yearFilter_low=2006,):

    schema = {
            "Condition": pl.Utf8,
            "Subgroup": pl.Utf8,
            "Year": pl.Int64,
            "Group": pl.Utf8,
            "Denominator_inc": pl.Float64,
            "Numerator_inc": pl.Float64,#pl.Int64,
            "Incidence": pl.Float64,
            "LowerCI_inc": pl.Float64,
            "UpperCI_inc": pl.Float64,
            "Denominator_prev": pl.Int64,
            "Numerator_prev": pl.Float64,#pl.Int64,
            "Prevalence": pl.Float64,
            "LowerCI_prev": pl.Float64,
            "UpperCI_prev": pl.Float64,
            }

    commonCols = tuple(["Condition",
                        "Subgroup",
                        "Year",
                        "Group",])
    incCols = commonCols + tuple(["Incidence",
                                  "LowerCI_inc",
                                  "UpperCI_inc",])
    prevCols = commonCols + tuple(["Prevalence",
                                  "LowerCI_prev",
                                  "UpperCI_prev",])

    dat = pl.read_csv(path_dat, schema=schema,)

    def formatDat(dat,
                  cols,
                  yearFilter_low,
                  suff):
        if suff=="inc":
            metric = "Incidence"
        elif suff=="prev":
            metric = "Prevalence"

        dat_ = (
                dat
                .select(pl.col(cols))
                .filter(
                    pl.col("Year") >= yearFilter_low
                    )
                .filter(
                    pl.col("Group").is_in([
                        "IMD_pracid",
                        "ETHNICITY",
                        "OVERALL",
                        ])
                    )
                .filter(
                    pl.col(metric).is_not_null()
                    )
                .rename(
                    {f"UpperCI_{suff}":"UpperCI", f"LowerCI_{suff}":"LowerCI",}
                    )
                .with_columns(
                    pl.col("Year").cast(pl.Utf8)
                    )
                .with_columns(
                    pl.col("Year").str.strptime(pl.Datetime, "%Y")
                    )
                .with_columns(
                    pl.col("Subgroup").str.replace_all("'", "")
                    )
                .filter(
                    pl.col("Subgroup") != "MissingImd"
                    )
                )

        return dat_

    dat_inc = formatDat(dat, incCols, yearFilter_low, "inc")
    dat_prev = formatDat(dat, prevCols, yearFilter_low, "prev")

    return dat_inc, dat_prev

#####################################################################
if DATA == "crude":
    dat_inc, dat_prev = read_dat_crude("out/Publish/crude.csv")
else:
    dat_inc, dat_prev = read_dat_dsr("out/Publish/DSR.csv")

dict_varsPlot = {
        "ethnicity": {"crude": "ETHNICITY", "dsr": "Ethnicity",},
        "deprivation": {"crude": "IMD_pracid", "dsr": "Deprivation",},
        "overall": {"crude": "OVERALL", "dsr": "Overall",},
        }

labels = tuple(dat_inc.get_column("Condition").unique().to_list())

files_out_label = []
for i_label, label in enumerate(labels):

    def getPng(traces, layout, label):
        global dpi
        global f
        global metric
        global DIR_OUT
        if metric == "Prevalence":
            metric_label = "Point Prevalence (per 100,000)"
        else:
            metric_label = "Incidence Rate (per 100,000 person years)"

        if layout is not None:
            layout["legend_yanchor"] = "top"
            layout["legend_xanchor"] = "left"
            layout["legend_y"] = 0.99
            layout["legend_x"] = 0.01
            layout["legend_title_text"] = ""
            layout["title_text"] = metric_label
            layout["margin"] = {'t':100, 'b':20, 'l':30, 'r':20,}
            layout["yaxis_title_standoff"] = 10
            layout["xaxis_title_standoff"] = 10
            layout["title_pad"] = {'t':20, 'b':20, 'l':30, 'r':20,}
            layout["plot_bgcolor"] = "white"
            layout["xaxis_gridcolor"] = "rgba(156, 156, 156, 0.7)"
            layout["yaxis_gridcolor"] = "rgba(156, 156, 156, 0.7)"



        plot = plt.graph_objects.Figure(
            data = traces,
            layout = layout
        )
        for i, x in enumerate(plot.layout.annotations):
            x.update(
                     font={"size": 40,
                           "family": f.dict_lay["font_family"]}
                     )

        plot.update_geos(fitbounds="locations", visible=False,
                          showframe=True,)
                         #bgcolor="rgba(189, 189, 189, 0.7)")


        plot.write_image(f"{DIR_OUT}IndivFigs/{label}.jpeg",
                         engine="kaleido",
                         width=8.3/2*dpi,
                         height=11.7/3*dpi,)


    files_out = []

    for i, page_ in enumerate(PAGES_CONFIG.keys()):
        page_ = PAGES_CONFIG[page_]

        plots = plt.subplots.make_subplots(
            cols=2, rows=2,
            vertical_spacing=0.1,
            horizontal_spacing=0.06,
            subplot_titles=page_["plt_titles"],
            specs=list(page_["spec_block"]),
            column_widths=page_["column_widths"],
            row_heights=page_["row_heights"],
            shared_xaxes=False,
            shared_yaxes=False,
        )

        f = ag.Visualisation(sf=2)
        f.ignore.append("Ireland")
        f.dict_lay["table_font_size"] = 26
        f.dict_lay["axes_tickfont_size"] = 12
        f.dict_lay["axes_title_font_size"] = 30
        f.dict_lay["axes_tickfont_size"] = 24
        f.dict_lay["background_opacity"] = 0
        f.dict_lay["background_fillcolor"] = "white"
        f.dict_lay["font_family"] = 'Helvetica, sans-serif'
        f.dict_lay["table_columnwidth"] = [1, 6]
        f.colours = [
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
        f.dict_lay["marker_size"] = 18
        f.dict_lay["line_width"] = 5
        f.dict_lay["error_y_thickness"] = 3
        f.dict_lay["x_axis_tickangle"] = 0
        f.dict_lay["table_height"] = 45
        f.dict_lay["x_axis_ticklabelstep"] = 2

        #f.dict_lay["x_axis_dtick"] = 1
        #f.dict_lay["x_axis_tick0"] = 2005
        #f.dict_lay["x_axis_autorange"] = False

        f.dict_lay["plot_bgcolor"] = "white"
        f.dict_lay["xaxis_gridcolor"] = "rgba(156, 156, 156, 0.7)"
        f.dict_lay["yaxis_gridcolor"] = "rgba(156, 156, 156, 0.7)"

        f.dict_lay["legend_font_size"] = 26

        ### PAGE 1
        if i==0:
        #####################################################################
            ### col1
            row = 1
            metric = "Prevalence"
            if metric == "Prevalence":
                metric_short = "Prev"
            elif metric == "Incidence":
                metric_short = "Inc"

            col = 1

            dat_ = (
                    dat_prev
                    .filter(pl.col("Condition")==label)
                    .filter(pl.col("Group") == dict_varsPlot["overall"][DATA])
                    )


            traces, layout = f.plot_scatter(dat_.to_pandas(),
                                            metric,
                                            "Year",
                                            c_name=None,
                                            withLine=True,
                                            is_errorY=True,
                                            cols_errorY=["LowerCI", "UpperCI"],
                                            legend=False,
                                            )
            #getPng(traces, layout, f"Overall_{metric}")

            for t in traces:
                plots.add_trace(t, row=row, col=col)

            if metric == "Prevalence":
                layout.yaxis.title.text = "Prevalence (per 100,000)"
            elif metric == "Incidence":
                layout.yaxis.title.text = "Incidence rate (per 100,000 person years)"

            plots.update_xaxes(layout.xaxis, col=col, row=row)
            plots.update_yaxes(layout.yaxis, col=col, row=row)

            plots.update_xaxes(title_standoff=title_standoff)
            plots.update_yaxes(title_standoff=title_standoff)

            #########################################################################
            row = 2
            metric = "Incidence"
            if metric == "Prevalence":
                metric_short = "Prev"
            elif metric == "Incidence":
                metric_short = "Inc"

            col = 1

            dat_ = (
                    dat_inc
                    .filter(pl.col("Condition")==label)
                    .filter(pl.col("Group") == dict_varsPlot["overall"][DATA])
                    )

            traces, layout = f.plot_scatter(dat_.to_pandas(),
                                            metric,
                                            "Year",
                                            c_name=None,
                                            withLine=True,
                                            is_errorY=True,
                                            cols_errorY=["LowerCI", "UpperCI"],
                                            legend=False,
                                            )
            #getPng(traces, layout, f"Overall_{metric}")

            for t in traces:
                plots.add_trace(t, row=row, col=col)

            if metric == "Prevalence":
                layout.yaxis.title.text = "Prevalence (per 100,000)"
            elif metric == "Incidence":
                layout.yaxis.title.text = "Incidence rate (per 100,000 person years)"
            plots.update_xaxes(layout.xaxis, col=col, row=row)
            plots.update_yaxes(layout.yaxis, col=col, row=row)

            plots.update_xaxes(title_standoff=title_standoff)
            plots.update_yaxes(title_standoff=title_standoff)

            ###################################################################
            ### col_2
            col = 2

            dat_table_ = (
                            dat_inc
                            .filter(pl.col("Condition")==label)
                            .rename({"UpperCI":"UpperCI_inc", "LowerCI":"LowerCI_inc",})
                            .join(
                                (
                                    dat_prev
                                    .filter(pl.col("Condition")==label)
                                    .rename({"UpperCI":"UpperCI_prev", "LowerCI":"LowerCI_prev",})
                                    ),
                                how="outer",
                                on=["Condition", "Subgroup", "Year",],
                                )
                            )

            dat_table_ = (
                    dat_table_
                    .filter(
                        pl.col("Group") == dict_varsPlot["overall"][DATA],
                        )
                    .with_columns(
                        pl.col(["Incidence",
                                "Prevalence",
                                "LowerCI_inc",
                                "UpperCI_inc",
                                "LowerCI_prev",
                                "UpperCI_prev",
                                ]).round(1)
                        )
                    .with_columns(
                        (pl.col("Year").dt.year()).alias("Year_Form"),
                        pl.concat_list(
                            pl.col("Incidence", "LowerCI_inc", "UpperCI_inc")
                            ).alias("Incidence_"),
                        pl.concat_list(
                            pl.col("Prevalence", "LowerCI_prev", "UpperCI_prev")
                            ).alias("Prevalence_")
                        )
                    .with_columns(
                        pl.col("Incidence_").map_elements(lambda x: f"{x[0]} ({x[1]}, {x[2]})"),
                        pl.col("Prevalence_").map_elements(lambda x: f"{x[0]} ({x[1]}, {x[2]})"),
                        )
                    )

            dat_table_ = dat_table_[["Year_Form", "Incidence_", "Prevalence_"]]
            #dat_table_inc = dat_table_[(dat_table_["Incidence_"].notnull())][["Year_Form", "Incidence_", group]]
            #dat_table_prev = dat_table_[(dat_table_["Prevalence_"].notnull())][["Year_Form", group, "Prevalence_"]]
            #dat_table_ = dat_table_inc.merge(dat_table_prev, how="left", on=["Year_Form", group])

            #dat_table_ = dat_table_[["Year_Form", f"{metric}_"]]
            #dat_table_ = dat_table_[dat_table_["Year_Form"]>2004]

            #dat_table_ = dat_table_.join(
            #        dat_table_[f"{metric}_"].str.split(
            #            " ", expand=True, n=1,
            #            )
            #        )
            #dat_table_ = dat_table_.loc[:,~(dat_table_.columns == f"{metric}_")]

            f.dict_lay["table_cells_align"] = ["left", "right", "left"]
            f.dict_lay["table_header_sizeMulti"] = 1.2
            f.dict_lay["table_header_height"] = 50
            f.dict_lay["table_cells_height"] = 40
            f.dict_lay["table_header_align"] = f.dict_lay["table_cells_align"]
            f.dict_lay["table_columnwidth"] = [0.2,1.05,0.4]

            plot_table = f.table_Incprev(dat_table_.select(pl.col(["Year_Form","Prevalence_"])).to_pandas(),
                                        formatData=False,
                                        columns=["Year", "Prevalence (95% CI)"],
                                         i_incidence=1, i_prevalence=1,
                                         )
            plots = plots.add_trace(plot_table, row=1, col=col)

            plot_table = f.table_Incprev(dat_table_.select(pl.col(["Year_Form","Incidence_"])).to_pandas(),
                                        formatData=False,
                                        columns=["Year", "Incidence (95% CI)"],
                                         i_incidence=1, i_prevalence=1,
                                         )
            plots = plots.add_trace(plot_table, row=2, col=col)

        ###PAGE 2
        if i==1:
        #####################################################################
            ### row1
            row = 1
            metric = "Prevalence"
            if metric == "Prevalence":
                metric_short = "Prev"
            elif metric == "Incidence":
                metric_short = "Inc"

            col = 1

            dat_ = (
                    dat_prev
                    .filter(pl.col("Condition")==label)
                    .filter(pl.col("Group") == dict_varsPlot["ethnicity"][DATA])
                    )

            if dat_.get_column(metric).max() is None or dat_.get_column(metric).max() < 1:
                ylims = [0, 1]
            else:
                ylims = None

            traces, layout = f.plot_scatter(dat_.to_pandas(),
                                            metric,
                                            "Year",
                                            c_name="Subgroup",
                                            withLine=True,
                                            is_errorY=True,
                                            cols_errorY=["LowerCI", "UpperCI"],
                                            useObjectColours=True,
                                            colourDictionaryId="Ethnicity",
                                            legend=True,
                                            ylims=ylims,
                                            )
            #getPng(traces, layout, f"Overall_{metric}")

            for t in traces:
                plots.add_trace(t, row=row, col=col)

            if metric == "Prevalence":
                layout.yaxis.title.text = "Prevalence (per 100,000)"
            elif metric == "Incidence":
                layout.yaxis.title.text = "Incidence rate (per 100,000 person years)"

            plots.update_xaxes(layout.xaxis, col=col, row=row)
            plots.update_yaxes(layout.yaxis, col=col, row=row)

            plots.update_xaxes(title_standoff=title_standoff)
            plots.update_yaxes(title_standoff=title_standoff)

            #########################################################################
            metric = "Incidence"
            if metric == "Prevalence":
                metric_short = "Prev"
            elif metric == "Incidence":
                metric_short = "Inc"

            col = 2

            dat_ = (
                    dat_inc
                    .filter(pl.col("Condition")==label)
                    .filter(pl.col("Group") == dict_varsPlot["ethnicity"][DATA])
                    )

            if dat_.get_column(metric).max() is None or dat_.get_column(metric).max() < 1:
                ylims = [0, 1]
            else:
                ylims = None

            traces, layout = f.plot_scatter(dat_.to_pandas(),
                                            metric,
                                            "Year",
                                            c_name="Subgroup",
                                            withLine=True,
                                            is_errorY=True,
                                            cols_errorY=["LowerCI", "UpperCI"],
                                            useObjectColours=True,
                                            colourDictionaryId="Ethnicity",
                                            legend=False,
                                            ylims=ylims,
                                            )
            #getPng(traces, layout, f"Overall_{metric}")

            for t in traces:
                plots.add_trace(t, row=row, col=col)

            if metric == "Prevalence":
                layout.yaxis.title.text = "Prevalence (per 100,000)"
            elif metric == "Incidence":
                layout.yaxis.title.text = "Incidence rate (per 100,000 person years)"

            plots.update_xaxes(layout.xaxis, col=col, row=row)
            plots.update_yaxes(layout.yaxis, col=col, row=row)

            plots.update_xaxes(title_standoff=title_standoff)
            plots.update_yaxes(title_standoff=title_standoff)

            ###################################################################
            ### row2
            ### col1
            row = 2
            metric = "Prevalence"
            if metric == "Prevalence":
                metric_short = "Prev"
            elif metric == "Incidence":
                metric_short = "Inc"

            col = 1

            dat_ = (
                    dat_prev
                    .filter(pl.col("Condition")==label)
                    .filter(pl.col("Group") == dict_varsPlot["deprivation"][DATA])
                    )

            if dat_.get_column(metric).max() is None or dat_.get_column(metric).max() < 1:
                ylims = [0, 1]
            else:
                ylims = None

            traces, layout = f.plot_scatter(dat_.to_pandas(),
                                            metric,
                                            "Year",
                                            c_name="Subgroup",
                                            withLine=True,
                                            is_errorY=True,
                                            cols_errorY=["LowerCI", "UpperCI"],
                                            useObjectColours=True,
                                            colourDictionaryId="Deprivation",
                                            legend=True,
                                            ylims=ylims,
                                            )
            #getPng(traces, layout, f"Overall_{metric}")

            for t in traces:
                plots.add_trace(t, row=row, col=col)

            if metric == "Prevalence":
                layout.yaxis.title.text = "Prevalence (per 100,000)"
            elif metric == "Incidence":
                layout.yaxis.title.text = "Incidence rate (per 100,000 person years)"

            plots.update_xaxes(layout.xaxis, col=col, row=row)
            plots.update_yaxes(layout.yaxis, col=col, row=row)

            plots.update_xaxes(title_standoff=title_standoff)
            plots.update_yaxes(title_standoff=title_standoff)

            #########################################################################
            metric = "Incidence"
            if metric == "Prevalence":
                metric_short = "Prev"
            elif metric == "Incidence":
                metric_short = "Inc"

            col = 2

            dat_ = (
                    dat_inc
                    .filter(pl.col("Condition")==label)
                    .filter(pl.col("Group") == dict_varsPlot["deprivation"][DATA])
                    )

            if dat_.get_column(metric).max() is None or dat_.get_column(metric).max() < 1:
                ylims = [0, 1]
            else:
                ylims = None

            traces, layout = f.plot_scatter(dat_.to_pandas(),
                                            metric,
                                            "Year",
                                            c_name="Subgroup",
                                            withLine=True,
                                            is_errorY=True,
                                            cols_errorY=["LowerCI", "UpperCI"],
                                            useObjectColours=True,
                                            colourDictionaryId="Deprivation",
                                            legend=False,
                                            ylims=ylims,
                                            )
            #getPng(traces, layout, f"Overall_{metric}")

            for t in traces:
                plots.add_trace(t, row=row, col=col)

            if metric == "Prevalence":
                layout.yaxis.title.text = "Prevalence (per 100,000)"
            elif metric == "Incidence":
                layout.yaxis.title.text = "Incidence rate (per 100,000 person years)"

            plots.update_xaxes(layout.xaxis, col=col, row=row)
            plots.update_yaxes(layout.yaxis, col=col, row=row)

            plots.update_xaxes(title_standoff=title_standoff)
            plots.update_yaxes(title_standoff=title_standoff)

            ###################################################################


        #######################################################################

        layout.xaxis = None
        layout.yaxis = None
        layout["legend_yanchor"] = "top"
        layout["legend_y"] = 0.95
        layout["legend_x"] = 1
        if i==0:
            layout["title_text"] = f"Standardised overall statistics of {label}"
        elif i==1:
            layout["title_text"] = f"Standardised grouped statistics of {label}"
        layout["title_xanchor"] = "left"
        layout["title_x"] = 0.03
        layout["title_yanchor"] = "top"
        layout["title_automargin"] = True
        layout["title_pad"] = {
                "t":40,
                "b":60,
                "l":0,
                "r":0,
                }

        #####################################################################
        for i_ann, x in enumerate(plots.layout.annotations):
            if i_ann%2 == 0:
                x.update(x=-0.05, y=x.y+0.008,
                         xanchor="left",
                         font={"size": 40,
                               "family": f.dict_lay["font_family"]})
            else:
                x.update(x=0.45, y=x.y+0.008,
                         xanchor="left",
                         font={"size":40,
                               "family": f.dict_lay["font_family"]})

        layout["margin_t"] = 100
        layout["margin_b"] = 100
        layout["margin_l"] = 150
        layout["margin_r"] = 150

        layout["legend_title_text"] = ""
        layout["legend_tracegroupgap"] = 700#350
        layout["plot_bgcolor"] = f.dict_lay["plot_bgcolor"]

        plots.update_layout(layout)

        file_name = f"{DIR_OUT}Summaries/{label}_overall_summary_{i}.pdf"
        files_out.append(file_name)
        plots.write_image(file_name,
                          engine="kaleido",
                          width=PAPER_DIM["a4"][0]*dpi,
                          height=PAPER_DIM["a4"][1]*dpi,
                          )
        ##########################################################
    out = Pdf.new()
    for file_name in files_out:
        with Pdf.open(file_name) as to_merge:
            out.pages.append(to_merge.pages[0])
        os.remove(file_name)

    outFile_ = f"{DIR_OUT}/{label}.pdf"
    files_out_label.append(f"{DIR_OUT}/{label}.pdf")
    out.save(f"{DIR_OUT}/{label}.pdf")
    out.close()

    print(f"Completed {label}")


out = Pdf.new()
for file_name in files_out_label:
    with Pdf.open(file_name) as to_merge:
        out.pages.append(to_merge.pages[0])
        out.pages.append(to_merge.pages[1])
    #os.remove(file_name)

out.save(f"{DIR_OUT}Summaries/Overall_summary.pdf")
out.close()
#Delete single page files

#add footer
from reportlab.pdfgen.canvas import Canvas
from pdfrw import PdfReader
from pdfrw.toreportlab import makerl
from pdfrw.buildxobj import pagexobj

def add_footer_to_pdf(input_file, footer_text):
    # Get pages
    reader = PdfReader(input_file)
    pages = [pagexobj(p) for p in reader.pages]


    # Compose new pdf
    canvas = Canvas(input_file)

    for page_num, page in enumerate(pages, start=1):

        # Add page
        canvas.setPageSize((page.BBox[2], page.BBox[3]))
        canvas.doForm(makerl(canvas, page))

        # Draw footer
        x = 150
        canvas.saveState()
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(0.5)
        #canvas.line(66, 78, page.BBox[2] - 66, 78)
        canvas.setFont('Times-Roman', 14)
        canvas.drawString(page.BBox[2]-x, 40, footer_text)
        canvas.restoreState()

        canvas.showPage()

    canvas.save()

add_footer_to_pdf(f"{DIR_OUT}Summaries/Overall_summary.pdf",
                  "doi:")
for file_name in files_out_label:
    add_footer_to_pdf(file_name,
                      "doi:")

print(f"Completed script")

