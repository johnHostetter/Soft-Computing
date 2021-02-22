#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:50:21 2021

@author: john
"""

import matplotlib.pyplot as plt

def subs(x):
    """
    Substitutes True values for 1.0, and substitutes
    False values for -1.0. Necessary for the rules in
    the APFRB to calculate the consequents correctly.

    Parameters
    ----------
    x : boolean
        A boolean describing whether the linguistic term
        is term- or term+. If the linguistic term is term+,
        then x is True. Else, the linguistic term is term-,
        and x is False.

    Returns
    -------
    float
        A float value to modify the a_i values to have
        the correct sign when calculating a rule's consequent.

    """
    if x:
        return 1.0
    else:
        return -1.0
    
def plot(title, x_lbl, y_lbl):
    """
    Handles the basic mechanics of plotting a graph
    such as assigning the x or y label, title, etc.
    Shows the plot after the function call is complete.

    Parameters
    ----------
    title : string
        Title of the plot.
    x_lbl : string
        Title of the x label.
    y_lbl : string
        Title of the y label.

    Returns
    -------
    None.

    """
    plt.title(title)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.show()       
    
def line(x, y, title, x_lbl, y_lbl):
    """
    

    Parameters
    ----------
    x : list
        A list containing the x values.
    y : list
        A list containing the y values.
    title : string
        Title of the plot.
    x_lbl : string
        Title of the x label.
    y_lbl : string
        Title of the y label.

    Returns
    -------
    None.

    """
    plt.plot(x, y)
    plot(title, x_lbl, y_lbl)
    
def bar(x, heights, title, x_lbl, y_lbl):
    """
    

    Parameters
    ----------
    x : list
        A list containing the x values.
    heights : list
        A list containing the heights of the bars.
    title : string
        Title of the plot.
    x_lbl : string
        Title of the x label.
    y_lbl : string
        Title of the y label.

    Returns
    -------
    None.

    """
    plt.bar(x, heights)
    plot(title, x_lbl, y_lbl)