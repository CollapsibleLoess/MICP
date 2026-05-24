import dearpygui.dearpygui as dpg

dpg.create_context()

with dpg.theme() as t:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       [18, 18, 30, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        [22, 28, 48, 255])
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg,        [22, 28, 48, 255])
        dpg.add_theme_color(dpg.mvThemeCol_Border,         [40, 50, 80, 180])
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        [28, 28, 46, 255])
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [36, 36, 60, 255])
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  [44, 44, 74, 255])
        dpg.add_theme_color(dpg.mvThemeCol_Button,         [15, 52, 96, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  [83, 52, 131, 255])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   [0, 160, 180, 255])
        dpg.add_theme_color(dpg.mvThemeCol_Text,           [230, 230, 240, 255])
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,   [120, 120, 140, 255])
        dpg.add_theme_color(dpg.mvThemeCol_Tab,            [22, 28, 48, 255])
        dpg.add_theme_color(dpg.mvThemeCol_TabHovered,     [15, 52, 96, 255])
        dpg.add_theme_color(dpg.mvThemeCol_TabActive,      [0, 160, 180, 255])
        dpg.add_theme_color(dpg.mvThemeCol_Separator,      [40, 50, 80, 180])
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        [15, 15, 28, 255])
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  [15, 52, 96, 255])
        dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg,      [15, 15, 28, 255])
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark,      [0, 200, 220, 255])
        dpg.add_theme_color(dpg.mvThemeCol_Header,         [15, 52, 96, 255])
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  [83, 52, 131, 255])
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,   [0, 160, 180, 255])
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
        dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 3)
dpg.bind_theme(t)

with dpg.theme(tag="_pt"):
    with dpg.theme_component(dpg.mvPlot):
        dpg.add_theme_color(dpg.mvPlotCol_PlotBg,       [12, 12, 22, 255])
        dpg.add_theme_color(dpg.mvPlotCol_PlotBorder,   [40, 50, 80, 180])
        dpg.add_theme_color(dpg.mvPlotCol_AxisText,     [140, 140, 160, 255])
        dpg.add_theme_color(dpg.mvPlotCol_AxisGrid,     [40, 50, 80, 100])
        dpg.add_theme_color(dpg.mvPlotCol_LegendBg,     [22, 28, 48, 220])
        dpg.add_theme_color(dpg.mvPlotCol_LegendBorder, [40, 50, 80, 180])
        dpg.add_theme_color(dpg.mvPlotCol_LegendText,   [210, 210, 220, 255])

dpg.create_viewport(width=1200, height=800, title="test5")

with dpg.window(tag="win"):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Open")
    with dpg.group(horizontal=True):
        with dpg.child_window(width=360, tag="left"):
            dpg.add_text("MICP Processor", color=[233, 69, 96])
            dpg.add_separator()
            dpg.add_text("Parameters", color=[0, 200, 220])
            dpg.add_input_float(label="Angle", default_value=130.0, width=110, format="%.4f", step=0)
            dpg.add_input_int(label="Window", default_value=7, width=110)
            dpg.add_button(label="Recalculate", width=-1, height=30)
            dpg.add_separator()
            dpg.add_text("Sample", color=[0, 200, 220])
            dpg.add_text("Load data...", color=[140, 140, 160])
        with dpg.child_window(tag="right"):
            with dpg.group(horizontal=True, parent="right"):
                dpg.add_text("Chart:", parent="right")
                dpg.add_combo(["A", "B"], default_value="A", tag="sel", width=200, parent="right")
            with dpg.group(tag="plot_area", parent="right"):
                with dpg.plot(tag="plt", height=-1, width=-1, no_title=True, parent="plot_area"):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                    dpg.add_plot_axis(dpg.mvXAxis, label="d (nm)", tag="xax", scale=1, invert=True)
                    yax = dpg.add_plot_axis(dpg.mvYAxis, label="Sat (%)", tag="yax")
                    dpg.add_line_series([10, 100, 1000, 10000], [0, 30, 70, 95], parent=yax, label="test")
            dpg.bind_item_theme("plt", "_pt")

with dpg.file_dialog(label="Open", directory_selector=False, show=False,
    callback=lambda s,a:None, tag="fd1", width=800, height=500):
    dpg.add_file_extension(".xlsx", label="Excel")

with dpg.file_dialog(label="Save", directory_selector=False, show=False,
    callback=lambda s,a:None, tag="fd2", width=800, height=500):
    dpg.add_file_extension(".xlsx", label="Excel")

with dpg.file_dialog(label="SaveJ", directory_selector=False, show=False,
    callback=lambda s,a:None, tag="fd3", width=800, height=500):
    dpg.add_file_extension(".json", label="JSON")

with dpg.file_dialog(label="Dir", directory_selector=True, show=False,
    callback=lambda s,a:None, tag="fd4", width=800, height=500):
    pass

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("win", True)
dpg.start_dearpygui()
dpg.destroy_context()
