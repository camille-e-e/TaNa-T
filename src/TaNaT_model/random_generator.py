"""
Random number generators.

Depending on the option defined in setup_constants, the rng can by numpy default_rng, or specifically low numbers from the same rng (which increases all probabiliies), or numbers can be read in from a text file.

Only the default rng has been setup so far.
"""

import numpy as np
#from geoTNM 
import TNM_constants as const

# default rng
if const.options.rng == "default":
    class rng:
        def seed(seed):
            """ 
            seed the rng and return that rng for re-use
            IN: seed (int)
            OUT: rng
            """
            return np.random.default_rng(seed)
        # def random():
        #     return np.random.default_rng()
        # def choice(list_in,p):
        #     return np.random.choice(list_in,p=p)

# define other rng
elif const.options.rng == "low-numbers":
    class rng:
        def seed(seed):
            np.random.seed(seed)
        def random(size=1):
            """Return 1 random float or an np.array of them """
            if size == 1:
                # return random float betwee 0 and 0.2
                return np.random.uniform(0,0.2)
            else: return np.random.uniform(0,0.2,size)
        def choice(list_in,p):
            """ choose from a weighted list """
            return np.random.choice(list_in,p=p)

elif const.options.rng == "read-in":
    class rng:
        def random():
            return "set this up"
        def choice(list_in,p):
            return "set this up"

