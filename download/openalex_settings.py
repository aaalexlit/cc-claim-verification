# Here we define the 3 concept groups we want to link by AND operators
concept_groups = [
    """ "climate" OR elevated temperature" OR 
    "ocean warming" OR "saline intrusion" OR 
    "environmental change" OR "global warming" OR     
    "global change" OR "greenhouse effect" OR 
    "snow cover" OR "extreme temperature" OR 
    "cyclone" OR "ocean acidification" OR "anthropogenic" 
    OR "sea level" OR "precipitation variability" 
    OR "precipitation change" OR "temperature impact" OR 
    "environmental variability" OR "weather pattern" OR 
    "weather factor" OR "increase in temperature") """,
    """ "species" OR
    "mortality" OR "health" OR
    "disease" OR "ecosystem" OR
    "mass balance" OR "flood" OR
    "drought" OR "disease" OR
    "adaptation" OR "malaria" OR "fire"
    OR "water scarcity" OR "water
    supply" OR "permafrost" OR
    "biological response" OR "food
    availability" OR "food security" OR
    "vegetation dynamic" OR "cyclone"
    OR "yield" OR "gender" OR
    "indigenous" OR "conflict" OR
    "inequality" OR "snow water
    equivalent" OR "surface temperature" OR
    "glacier melt" OR
    "glacier mass" OR
    "coastal erosion" OR
    "glacier retreat" OR
    "rainfall reduction" OR
    "reduction in rainfall" OR
    "coral stress" OR "precipitation
    increase" OR "precipitation decrease"
    OR "river flow" """,
    """ "recent" OR "current" OR "modern" OR "observation" OR
    "observed" OR "observable"
    "evidence" OR "past" OR "local" OR "regional" OR 
    "significant" OR "driver" OR "driving" OR 
    "response" OR "were responsible" OR "was responsible" OR 
    "exhibited" OR "witnessed" OR "attribution" OR
    "attributed" OR "attributable" OR 
    "has increased" OR "has decreased" OR 
    "historic" OR "correlation" OR "evaluation" """,
]

keep_fields = [
    "id",
    "doi",
    "title",
    "publication_year",
]
