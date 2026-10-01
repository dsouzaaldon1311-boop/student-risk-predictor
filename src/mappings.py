# UCI codebook mappings — converts raw numeric codes into human-readable labels
# Source: Realinho et al. (2022), Predicting Student Dropout and Academic Success

MARITAL_STATUS = {
    1: "Single", 2: "Married", 3: "Widower",
    4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
}

APPLICATION_MODE = {
    1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
    3: "1st phase - special contingent (Azores)", 4: "Holders of other higher courses",
    5: "Ordinance No. 854-B/99", 6: "International student (bachelor)",
    7: "1st phase - special contingent (Madeira)", 8: "2nd phase - general contingent",
    9: "3rd phase - general contingent", 10: "Ordinance 533-A/99, b2 (Different Plan)",
    11: "Ordinance 533-A/99, b3 (Other Institution)", 12: "Over 23 years old",
    13: "Transfer", 14: "Change in course", 15: "Technological specialization diploma",
    16: "Change in institution/course", 17: "Short cycle diploma holders",
    18: "Change in institution/course (International)"
}

COURSE = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equiniculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening)"
}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education", 2: "Higher ed - bachelor's", 3: "Higher ed - degree",
    4: "Higher ed - master's", 5: "Higher ed - doctorate", 6: "Frequency of higher ed",
    7: "12th year - not completed", 8: "11th year - not completed",
    9: "Other - 11th year", 10: "10th year of schooling", 11: "10th year - not completed",
    12: "Basic ed 3rd cycle", 13: "Basic ed 2nd cycle", 14: "Technological specialization",
    15: "Higher ed - degree (1st cycle)", 16: "Professional higher technical course",
    17: "Higher ed - master's (2nd cycle)"
}

PARENT_QUALIFICATION = {
    1: "Secondary Ed (12th Year)", 2: "Higher Ed - bachelor's", 3: "Higher Ed - degree",
    4: "Higher Ed - master's", 5: "Higher Ed - doctorate", 6: "Frequency of Higher Ed",
    9: "12th Year - not completed", 10: "11th Year - not completed", 11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling", 14: "10th Year of Schooling",
    18: "General commerce course", 19: "Basic Ed 3rd Cycle (9th/10th/11th Year)",
    20: "Complementary High School Course", 22: "Technical-professional course",
    25: "Complementary HS - not concluded", 26: "7th year of schooling",
    27: "2nd cycle general high school", 29: "9th Year - not completed",
    30: "8th year of schooling", 31: "General Admin and Commerce Course",
    33: "Supplementary Accounting/Admin", 34: "Unknown",
    35: "Cannot read or write", 36: "Can read, no 4th year",
    37: "Basic ed 1st cycle (4th/5th year)", 38: "Basic Ed 2nd Cycle (6th/7th/8th Year)",
    39: "Technological specialization", 40: "Higher ed - degree (1st cycle)",
    41: "Specialized higher studies", 42: "Professional higher technical course",
    43: "Higher Ed - master's (2nd cycle)", 44: "Higher Ed - doctorate (3rd cycle)"
}

# Shared by Mother's and Father's occupation columns
PARENT_OCCUPATION = {
    1: "Student", 2: "Legislative/Executive/Director", 3: "Intellectual/Scientific Specialist",
    4: "Intermediate Technician", 5: "Administrative Staff",
    6: "Personal Services/Security/Sales", 7: "Farmer/Agriculture Worker",
    8: "Skilled Industry/Construction Worker", 9: "Machine Operator/Assembler",
    10: "Unskilled Worker", 11: "Armed Forces", 12: "Other", 13: "(blank)",
    14: "Armed Forces Officer", 15: "Armed Forces Sergeant", 16: "Other Armed Forces",
    17: "Admin/Commercial Services Director", 18: "Hotel/Catering/Trade Director",
    19: "Physical Science/Engineering Specialist", 20: "Health Professional",
    21: "Teacher", 22: "Finance/Accounting Specialist", 23: "Science/Engineering Technician",
    24: "Intermediate Health Professional", 25: "Legal/Social/Sports Technician",
    26: "IT Technician", 27: "Office Worker/Secretary", 28: "Data/Accounting Operator",
    29: "Other Admin Support", 30: "Personal Service Worker", 31: "Seller",
    32: "Personal Care Worker", 33: "Protection/Security Personnel",
    34: "Market-oriented Farmer", 35: "Subsistence Farmer/Fisher",
    36: "Skilled Construction Worker", 37: "Metalworking Worker",
    38: "Electrician/Electronics Worker", 39: "Food/Wood/Clothing Industry Worker",
    40: "Fixed Plant/Machine Operator", 41: "Assembly Worker",
    42: "Vehicle Driver", 43: "Unskilled Agriculture Worker",
    44: "Unskilled Industry/Construction Worker", 45: "Meal Prep Assistant",
    46: "Street Vendor/Service"
}

GENDER = {1: "Male", 0: "Female"}
ATTENDANCE = {1: "Daytime", 0: "Evening"}
YES_NO = {1: "Yes", 0: "No"}  # Displaced, Educational special needs, Debtor,
                               # Tuition fees up to date, Scholarship holder, International

# Maps each dataframe column name to the dictionary that decodes it
COLUMN_MAPPINGS = {
    "Marital status": MARITAL_STATUS,
    "Application mode": APPLICATION_MODE,
    "Course": COURSE,
    "Previous qualification": PREVIOUS_QUALIFICATION,
    "Mother's qualification": PARENT_QUALIFICATION,
    "Father's qualification": PARENT_QUALIFICATION,
    "Mother's occupation": PARENT_OCCUPATION,
    "Father's occupation": PARENT_OCCUPATION,
    "Gender": GENDER,
    "Daytime/evening attendance": ATTENDANCE,
    "Displaced": YES_NO,
    "Educational special needs": YES_NO,
    "Debtor": YES_NO,
    "Tuition fees up to date": YES_NO,
    "Scholarship holder": YES_NO,
    "International": YES_NO,
}


def decode_value(column_name, value):
    """Return a human-readable label for a raw coded value, or a cleaned raw value if unmapped."""
    mapping = COLUMN_MAPPINGS.get(column_name)
    if mapping is None:
        try:
            if float(value).is_integer():
                return int(value)
        except (ValueError, TypeError):
            pass
        return value
    return mapping.get(int(value), value)