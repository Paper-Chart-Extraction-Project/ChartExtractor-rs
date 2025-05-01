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
    /// Oxygen saturation.
    spo2: HashMap<String, u32>,
    /// End tidal carbon dioxide.
    etco2: HashMap<String, u32>,
    /// Fraction of inspired oxygen.
    fio2: HashMap<String, u32>,
    temperature: HashMap<String, f32>,
    tidal_volume: HashMap<String, u32>,
    respiratory_rate: HashMap<String, u32>,
    urine_output: HashMap<String, u32>,
    blood_loss: HashMap<String, u32>,
    /// The endotracheal tube size.
    ett_n: f32
}

/// A stuct containing all of the preoperative/postoperative chart's data.
struct PreoperativePostoperativeChart { }

struct Chart {
    intraoperative_charts: Vec<IntraoperativeChart>,
    preoperative_postoperative_chart: PreoperativePostoperativeChart
}
