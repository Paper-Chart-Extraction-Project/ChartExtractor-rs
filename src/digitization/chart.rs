use std::collections::HashMap;


/// Hour and minute.
struct Time ( u32, u32 );

/// An enum for single digit positive whole numbers.
enum SingleDigit {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
}

/// A drug or fluid code is a three digit number.
struct Code (SingleDigit, SingleDigit, SingleDigit);

/// Contains the Code for the drug or fluid, along with a HashMap mapping the
/// timestamp to the dose.
struct DosingRecord ( Code, HashMap<String, u32> );

/// Contains all 9 rows of the medications section.
struct MedicationSection ( 
    Option<DosingRecord>, // always propofol.
    Option<DosingRecord>, // always rocuronium.
    Option<DosingRecord>, // always fentanyl.
    Option<DosingRecord>,
    Option<DosingRecord>,
    Option<DosingRecord>,
    Option<DosingRecord>,
    Option<DosingRecord>,
    Option<DosingRecord>
);

/// Contains the 2 rows of the fluid/blood product section.
struct FluidBloodProductSection ( Option<DosingRecord>, Option<DosingRecord> );

/// A struct containing all of the intraoperative chart's data.
struct IntraoperativeChart {
    /// Which intraoperative page we are on. Some surgeries span multiple pages.
    page_num: u32,
    anesthesia_start: Option<Time>,
    anesthesia_end: Option<Time>,
    surgery_start: Option<Time>,
    surgery_end: Option<Time>,
    medications: MedicationSection,
    inhaled_volatile_gas: HashMap<String, u32>,
    fluid_and_blood_products: FluidBloodProductSection,
    checkboxes: HashMap<String, bool>,
    systolic_bp: HashMap<String, u32>,
    diastolic_bp: HashMap<String, u32>,
    heart_rate: HashMap<String, u32>,
    oxygen_saturation: HashMap<String, u32>,
    end_tidal_carbon_dioxide: HashMap<String, u32>,
    fraction_of_inspired_oxygen: HashMap<String, u32>,
    temperature: HashMap<String, f32>,
    tidal_volume: HashMap<String, u32>,
    respiratory_rate: HashMap<String, u32>,
    urine_output: HashMap<String, u32>,
    blood_loss: HashMap<String, u32>,
    endotracheal_tube_size: f32
}

/// The vitals
struct Vitals {
    systolic: u32,
    diastolic: u32,
    heart_rate: u32,
    respiratory_rate: u32,
    oxygen_saturation: u32
}

/// A stuct containing all of the preoperative/postoperative chart's data.
struct PreoperativePostoperativeChart {
    time_of_assessment_day: u32,
    time_of_assessment_month: u32,
    time_of_assessment_year: u32,
    time_of_assessment_hour: u32,
    time_of_assessment_minute: u32,
    checkboxes: HashMap<String, bool>,
    age: u32,
    height: u32,
    weight: u32,
    preoperative_vitals: Vitals,
    postoperative_vitals: Vitals,
    hemoglobin: f32,
    hematocrit: f32,
    platelets: u32,
    sodium: u32,
    potassium: f32,
    chloride: u32,
    urea: f32,
    creatinine: f32,
    calcium: f32,
    magnesium: f32,
    phosphate: f32,
    albumin: u32,
    aldrete_score: u32
}

struct Chart {
    intraoperative_charts: Vec<IntraoperativeChart>,
    preoperative_postoperative_chart: PreoperativePostoperativeChart
}
