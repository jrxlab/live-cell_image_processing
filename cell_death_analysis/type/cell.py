from __future__ import annotations


class cell_t:
    
    def __init__(self) -> None:
        #
        self.uid = -1  # 0 and up: unique ID within its frame
        # It corresponds to its position in its frame list of cells

        self.time_point = -1  # 0 and up: time point of the frame it belongs to
        self.position = ()  # (row, col): position of its centroid in the frame (not necessarily integers)
        self.pixels = None  # As returned by np_.nonzero: list of its composing pixels

        self.features = {}  # Dictionary of its feature values

    @classmethod
    
    def WithProperties(
        cls, uid: int, time_point: int, position: tuple, pixels: tuple
    ) -> cell_t:
        
        """
        WithProperties method
        Initiates the cell properties
        """
        instance = cls()

        instance.uid = uid # unique identifier 
        instance.time_point = time_point # time point
        instance.position = position # centroid
        instance.pixels = pixels # cell containing pixels

        return instance
    

    def __str__(self):
        
        """
        __str__ method
        Returns a string containing all the informations about each cell
        (unique identifier, time point, centroide coordinates)
        """
        return (
            f"{self.uid} " # unique identifier 
            f"in {self.time_point} " # time point
            f"@ {self.position[0]}x{self.position[1]}" # position X x Y
        )
