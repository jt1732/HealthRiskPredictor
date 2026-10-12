## 1. Row removal
Drop duplicate rows (ignoring `id`), Duplicates give those patients extra weight  
Keep height 130 to 210 cm; Covers essentially all adults  
Keep weight ≥ 40 kg 11 to 39 kg is implausible for an adult  
Keep BMI 12 to 80 BMI < 12 is incompatible with life, BMI > 80 is an extreme outlier.  
Keep ap_hi 50 to 250 Removes negatives and typing errors  
Keep ap_lo 30 to 200 Removes negatives and typing errors like 1000  
Keep ap_lo < ap_hi Diastolic can't be equal to or higher than systolic  
  
## 2. Not used as cleaning rules
An age filter. The youngest patient is 29.7, so there are no under-18s  

## 3. Feature selection
id Drop as it's an identifier, not a feature  
age Convert days to years (/365.25)  
bmi Add: `weight / (height / 100) ** 2` known risk factor   

