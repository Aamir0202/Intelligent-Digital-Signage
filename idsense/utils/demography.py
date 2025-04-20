def get_age_group(age):
    """Determine the age group category based on the given age."""

    if 7 <= age <= 12:
        return "7_12"
    elif 13 <= age <= 19:
        return "13_19"
    elif 20 <= age <= 35:
        return "20_35"
    elif 36 <= age <= 50:
        return "36_50"
    elif 51 <= age <= 65:
        return "51_65"
    elif age >= 66:
        return "66_plus"
    else:
        return "below_7"
