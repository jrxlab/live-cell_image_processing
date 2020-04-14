from cv2_rolling_ball import subtract_background_rolling_ball



def ContrastNormalized(image): 
    """
    Background subtraction method
    
    Subtracts background around each pixel over a large ball
    """
    
    img, background = subtract_background_rolling_ball(image, 86, light_background=False,
                                                       use_paraboloid=False, do_presmooth=False)
    
    return img