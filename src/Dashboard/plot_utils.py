# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

def update_fig_centered_title(fig, title):
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )