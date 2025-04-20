import random

ADS = {
    "man": {
        "7_12": [
            "action_toy_commercials",
            "educational_cartoons",
            "sports_gear_for_kids",
        ],
        "13_19": ["gaming_consoles", "sports_shoes", "gadget_reviews"],
        "20_35": ["smartphones", "fitness_equipment", "travel_packages"],
        "36_50": [
            "cars_and_suvs",
            "financial_investment_services",
            "health_supplements",
        ],
        "51_65": [
            "retirement_planning_services",
            "medical_equipment",
            "adventure_travel_for_seniors",
        ],
        "66_plus": ["healthcare_products", "memory_aids", "wellness_books"],
    },
    "woman": {
        "7_12": ["dollhouse_playsets", "craft_kits", "educational_books"],
        "13_19": ["beauty_products", "fashion_accessories", "art_supplies"],
        "20_35": [
            "fashion_trends",
            "skincare_products",
            "career_development_workshops",
        ],
        "36_50": [
            "home_decor_ideas",
            "health_and_wellness_programs",
            "parenting_resources",
        ],
        "51_65": [
            "lifestyle_blogs",
            "wellness_retreats",
            "financial_planning_for_women",
        ],
        "66_plus": ["gardening_tools", "comfort_wear", "hobby_classes"],
    },
}


def recommend_ad(gender, age_group):
    """Recommend an advertisement based on gender and age group."""

    try:
        return random.choice(ADS[gender][age_group])
    except KeyError:
        return "miscellaneous"
