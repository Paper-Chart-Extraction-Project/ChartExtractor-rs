use std::collections::HashMap;


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
    Option<DosingRecord>,
    Option<DosingRecord>,
    Option<DosingRecord>,
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
    medications: Vec<DosingRecord>
}

/// A stuct containing all of the preoperative/postoperative chart's data.

struct Chart {
    intraoperative_charts: Vec<IntraoperativeChart>,
    preoperative_postoperative_chart: PreoperativePostoperativeChart
}
