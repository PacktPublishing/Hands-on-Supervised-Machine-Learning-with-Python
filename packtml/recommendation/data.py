# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

__all__ = [
    'get_completely_fabricated_ratings_data'
]


def get_completely_fabricated_ratings_data():
    """Disclaimer: this is a made-up data set.

    Get a ratings data set for use with one of the packtml recommenders.
    This data set is a completely made-up ratings matrix consisting of
    cult classics, all of which are awesome (seriously, if there are any
    you haven't seen, you should).

    (Please
                    don't
                sue

                             me......)

    The data contains 5 users and 15 items (movies). Movies:

      0)  Ghost Busters
      1)  Ghost Busters 2
      2)  The Goonies
      3)  Big Trouble in Little China
      4)  The Rocky Horror Picture Show
      5)  A Clockwork Orange
      6)  Pulp Fiction
      7)  Bill & Ted's Excellent Adventure
      8)  Weekend at Bernie's
      9)  Dumb and Dumber
      10) Clerks
      11) Jay & Silent Bob Strike Back
      12) Tron
      13) Total Recall
      14) The Princess Bride

    Notes
    -----
    Seriously, I fabricated all of these ratings semi-haphazardly. Don't
    take this as me bashing any movies.
    """
    return (np.array([
        # user 0 is a classic 30-yo millennial who is nostalgic for the 90s
        [5.0, 3.5, 5.0, 0.0, 0.0, 0.0, 4.5, 3.0,
         0.0, 2.5, 4.0, 4.0, 0.0, 1.5, 3.0],

        # user 1 is a 40-yo who only likes action
        [1.5, 0.0, 0.0, 1.0, 0.0, 4.0, 5.0, 0.0,
         2.0, 0.0, 3.0, 3.5, 0.0, 4.0, 0.0],

        # user 2 is a 12-yo whose parents are strict about what she watches.
        [4.5, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0,
         3.5, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0],

        # user 3 has just about seen it all, and doesn't really care for
        # the goofy stuff. (but seriously, who rates the Goonies 2/5???)
        [2.0, 1.0, 2.0, 1.0, 2.5, 4.5, 4.5, 0.5,
         1.5, 1.0, 2.0, 2.5, 3.5, 3.5, 2.0],

        # user 4 has just opened a netflix account and hasn't had a chance
        # to watch too much
        [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.5, 4.0, 0.0, 0.0],
    ]), np.array(["Ghost Busters", "Ghost Busters 2",
                  "The Goonies", "Big Trouble in Little China",
                  "The Rocky Horror Picture Show", "A Clockwork Orange",
                  "Pulp Fiction", "Bill & Ted's Excellent Adventure",
                  "Weekend at Bernie's", "Dumb and Dumber", "Clerks",
                  "Jay & Silent Bob Strike Back", "Tron", "Total Recall",
                  "The Princess Bride" ]))
