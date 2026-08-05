def HER2_status_simple():
    """
    Function to determine the HER2 status based on IHC results only.
    """

    # Input IHC values
    ihc_result = input("Enter IHC result (0, 1+, 2+, 3+): ")

    # Determine HER2 status based on IHC results
    if ihc_result == "3+":
        return "HER2 Positive"
    else:
        return "HER2 Negative"


def HER2_status_full():
    """
    Function to determine HER2 status based on IHC and FISH results.
    Returns a string indicating the HER2 status.
    """
    # Example input values for IHC and FISH results
    ihc_result = input("Enter IHC result (0, 1+, 2+, 3+): ")
    fish_result = input("Enter FISH result (positive, negative, equivocal): ")

    # Determine HER2 status based on IHC and FISH results
    if ihc_result == "3+":
        return "HER2 Positive"
    elif ihc_result == "0" or ihc_result == "1+":
        return "HER2 Negative"
    elif ihc_result == "2+":
        if fish_result.lower() == "positive":
            return "HER2 Positive"
        elif fish_result.lower() == "negative":
            return "HER2 Negative"
        else:
            return "HER2 Equivocal"
    else:
        return "Invalid IHC result"