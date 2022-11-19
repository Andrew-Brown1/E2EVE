"""
utils functions for the demo
"""

def assert_region_width(edit_region):
    # checking whether the user input for edit_region is within bounds
    
    if (not ((edit_region <= 0.6) and (edit_region >= 0.4))) or not isinstance(edit_region, float):
        
        raise Exception("edit region width must be float between 0.4 and 0.6")

def assert_region_placement(region_placement):
    # checking whether the user input for region placement is in correct format
    
    if not isinstance(region_placement,tuple):
        raise Exception("edit region placement must be tuple with two values between 0 and 255")
    
    if not (region_placement[0] >= 0 and region_placement[0] < 256):
        raise Exception("edit region placement must be tuple with two values between 0 and 255")
    
    if not (region_placement[1] >= 0 and region_placement[1] < 256):
        raise Exception("edit region placement must be tuple with two values between 0 and 255")
    
    if not (len(region_placement) == 2):
        raise Exception("edit region placement must be tuple with two values between 0 and 255")
        