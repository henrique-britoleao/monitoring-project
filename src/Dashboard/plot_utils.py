# -*- coding: utf-8 -*-

#####  Set Logger  #####
from src.utils.loggers import MainLogger
logger = MainLogger.getLogger(__name__)

def update_fig_centered_title(fig, title):
    """Function to center the figure title

    Args:
        fig (Figure): plotly figure
        title (str): title to display
    """
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )