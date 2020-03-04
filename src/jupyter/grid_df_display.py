# Source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side/50899244#50899244
import pandas as pd
from IPython.display import display,HTML

def grid_df_display(list_dfs, rows = 2, cols=3, fill = 'cols'):
    if fill not in ['rows', 'cols']: print("grid_df_display() - fill must be one of: 'rows', 'cols'")

    html_table = "<table style='width:100%; border:0px'>{content}</table>"
    html_row   = "<tr style='border:0px'>{content}</tr>"
    html_cell  = "<td style='width:{width}%;vertical-align:top;border:0px'>{{content}}</td>"
    html_cell  = html_cell.format(width=100/cols)

    cells = [ html_cell.format(content=df.to_html()) for df in list_dfs[:rows*cols] ]
    cells += cols * [html_cell.format(content="")] # pad

    if fill == 'rows':   # fill in rows first (first row: 0,1,2,... col-1)
        grid = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,rows*cols,cols)]
    elif fill == 'cols': # fill columns first (first column: 0,1,2,..., rows-1)
        grid = [ html_row.format(content="".join(cells[i:rows*cols:rows])) for i in range(0,rows)]
    else:
        grid = []

    # noinspection PyTypeChecker
    display(HTML(html_table.format(content="".join(grid))))

    # add extra dfs to bottom
    [display(list_dfs[i]) for i in range(rows*cols,len(list_dfs))]


if __name__ == "main":
    list_dfs = []
    list_dfs.extend((pd.DataFrame(2*[{"x":"hello"}]),
                     pd.DataFrame(2*[{"x":"world"}]),
                     pd.DataFrame(2*[{"x":"gdbye"}])))

    grid_df_display(3*list_dfs)