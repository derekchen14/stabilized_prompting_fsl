import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_break = 5

metric_by_task = {
    'clc': 'accuracy',
    'tlc': 'accuracy',
    'dst': 'f1_score',
    'rg': 'bow_similarity',
    'ir': 'recall@5'
}

STOP_TOKENS = ['done', 'exit', 'logout', 'finish', 'stop']

DATASETS = {
    'abcd': 'Action-Based Conversations Dataset',
    'dstc': 'Dialogue State Tracking Challenge 2',
    'gsim': 'Google Simulated Dialogue',
    'mwoz': 'MultiWoz 2.2',
    'sgd': 'Schema Guided Dialogue',
    'tt': 'TicketTalk'
}

CHECKPOINTS = {
    't5': {
        'small': 't5-small', 
        'medium': 't5-base',
        'large': 't5-11b' },
    'gpt': {
        'small': 'gpt2',
        'medium': 'gpt2-large',
        'large': 'EleutherAI/gpt-j-6B'},
    'bart': {
        'small': 'facebook/bart-base',
        'medium': 'facebook/bart-large',
        'large': 'facebook/bart-xlarge'}
}

GENERAL_TYPO = {
    # type
    "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "churches":"church",
    "mutiple sports":"multiple sports", "sports":"multiple sports", "mutliple sports":"multiple sports",
    "swimmingpool":"swimming pool", "concerthall":"concert hall", "concert":"concert hall",
    "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture",
    "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", 
    # area
    "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north",
    "cen":"centre", "east side":"east", "east area":"east", "west part of town":"west", "ce":"centre",
    "town center":"centre", "centre of cambridge":"centre", "city center":"centre", "the south":"south",
    "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north",
    "centre of town":"centre", "cb30aq": "none",
    # price
    "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate",
    # day
    "next friday":"friday", "monda": "monday",
    # names
    "catherine s": "catherines",
    # parking
    "free parking":"free",
    # internet
    "free internet":"yes", "y":"yes",
    # star
    "4 star":"4", "4 stars":"4", "0 star rarting":"<none>", "3 .":"3",
    # others
    "n":"no", "does not":"no", "does not care":"any", "dontcare": "any",
    "not men":"<none>", "not":"<none>", "art":"<none>", "not mendtioned":"<none>", "fun":"<none>",
}

DOMAIN_SLOTS_MWOZ = {
  "restaurant": ["area", "people", "day", "time", "food", "name", "pricerange"],
  "taxi": ["arriveby", "destination", "departure", "leaveat"],
  "train": ["arriveby", "people", "day", "destination", "departure", "leaveat"],
  "hotel": ["area", "day", "stay", "people", "internet", "name", "parking", "pricerange", "stars", "type"],
  "attraction": ["area", "name", "type"]
}

DOMAIN_SLOTS_SGD = {
  "Alarm_1": [ "alarm_time", "alarm_name", "new_alarm_time", "new_alarm_name"],
  "Buses_3": [
    "from_city",
    "to_city",
    "from_station",
    "to_station",
    "departure_date",
    "departure_time",
    "price",
    "additional_luggage",
    "num_passengers",
    "category"],
  "Events_3": [
    "event_type",
    "event_name",
    "date",
    "time",
    "number_of_tickets",
    "price_per_ticket",
    "city",
    "venue",
    "venue_address"
  ],
  "Flights_4": [
    "number_of_tickets",
    "seating_class",
    "origin_airport",
    "destination_airport",
    "departure_date",
    "return_date",
    "is_nonstop",
    "outbound_departure_time",
    "outbound_arrival_time",
    "inbound_arrival_time",
    "inbound_departure_time",
    "price",
    "airlines"
  ],
  "Homes_2": [
    "intent",
    "area",
    "address",
    "property_name",
    "phone_number",
    "has_garage",
    "in_unit_laundry",
    "price",
    "visit_date",
    "number_of_beds",
    "number_of_baths"
  ],
  "Hotels_2": [
    "where_to",
    "number_of_adults",
    "check_in_date",
    "check_out_date",
    "rating",
    "address",
    "phone_number",
    "total_price",
    "has_laundry_service"
  ],
  "Hotels_4": [
    "location",
    "number_of_rooms",
    "check_in_date",
    "stay_length",
    "star_rating",
    "place_name",
    "street_address",
    "phone_number",
    "price_per_night",
    "smoking_allowed"
  ],
  "Media_3": [
    "title",
    "genre",
    "subtitle_language",
    "starring"
  ],
  "Messaging_1": [
    "location",
    "contact_name"
  ],
  "Movies_1": [
    "price",
    "number_of_tickets",
    "show_type",
    "theater_name",
    "show_time",
    "show_date",
    "genre",
    "street_address",
    "location",
    "movie_name"
  ],
  "Movies_3": [
    "movie_title",
    "genre",
    "percent_rating",
    "cast",
    "directed_by"
  ],
  "Music_3": [
    "track",
    "artist",
    "album",
    "genre",
    "year",
    "device"
  ],
  "Payment_1": [
    "payment_method",
    "amount",
    "receiver",
    "private_visibility"
  ],
  "RentalCars_3": [
    "car_type",
    "car_name",
    "pickup_location",
    "start_date",
    "pickup_time",
    "city",
    "end_date",
    "price_per_day",
    "add_insurance"
  ],
  "Restaurants_2": [
    "restaurant_name",
    "date",
    "time",
    "has_seating_outdoors",
    "has_vegetarian_options",
    "phone_number",
    "rating",
    "address",
    "number_of_seats",
    "price_range",
    "location",
    "category"
  ],
  "RideSharing_2": [
    "destination",
    "ride_type",
    "ride_fare",
    "wait_time",
    "number_of_seats"
  ],
  "Services_1": [
    "stylist_name",
    "phone_number",
    "average_rating",
    "is_unisex",
    "street_address",
    "city",
    "appointment_date",
    "appointment_time"
  ],
  "Services_4": [
    "therapist_name",
    "phone_number",
    "address",
    "city",
    "appointment_date",
    "appointment_time",
    "type"
  ],
  "Trains_1": [
    "from",
    "to",
    "from_station",
    "to_station",
    "date_of_journey",
    "journey_start_time",
    "total",
    "number_of_adults",
    "class",
    "trip_protection"
  ],
  "Travel_1": [
    "location",
    "attraction_name",
    "category",
    "phone_number",
    "free_entry",
    "good_for_kids"
  ],
  "Weather_1": [
    "precipitation",
    "humidity",
    "wind",
    "temperature",
    "city",
    "date"
  ],
  "Banks_2": [
    "account_type",
    "recipient_account_type",
    "account_balance",
    "transfer_amount",
    "recipient_name",
    "transfer_time"
  ],
  "Buses_1": [
    "from_location",
    "to_location",
    "from_station",
    "to_station",
    "leaving_date",
    "leaving_time",
    "fare",
    "travelers",
    "transfers"
  ],
  "Events_1": [
    "category",
    "subcategory",
    "event_name",
    "date",
    "time",
    "number_of_seats",
    "city_of_event",
    "event_location",
    "address_of_location"],
  "Flights_3": [
    "passengers",
    "flight_class",
    "origin_city",
    "destination_city",
    "origin_airport_name",
    "destination_airport_name",
    "departure_date",
    "return_date",
    "number_stops",
    "outbound_departure_time",
    "outbound_arrival_time",
    "inbound_arrival_time",
    "inbound_departure_time",
    "price",
    "number_checked_bags",
    "airlines",
    "arrives_next_day"],
  "Homes_1": [
    "area",
    "address",
    "property_name",
    "phone_number",
    "furnished",
    "pets_allowed",
    "rent",
    "visit_date",
    "number_of_beds",
    "number_of_baths"],
  "Hotels_1": [
    "destination",
    "number_of_rooms",
    "check_in_date",
    "number_of_days",
    "star_rating",
    "hotel_name",
    "street_address",
    "phone_number",
    "price_per_night",
    "has_wifi"],
  "Media_2": [
    "movie_name",
    "genre",
    "subtitle_language",
    "director",
    "actors",
    "price"],
  "Movies_2": [
    "title",
    "genre",
    "aggregate_rating",
    "starring",
    "director"],
  "Music_1": [
    "song_name",
    "artist",
    "album",
    "genre",
    "year",
    "playback_device"],
  "RentalCars_1": [
    "type",
    "car_name",
    "pickup_location",
    "pickup_date",
    "pickup_time",
    "pickup_city",
    "dropoff_date",
    "total_price"],
  "RideSharing_1": ["destination", "shared_ride", "ride_fare", "approximate_ride_duration", "number_of_riders"],
  "Banks_1": ["account_type", "recipient_account_type", "balance", "amount", "recipient_account_name"],
  "Buses_2": [
    "origin",
    "destination",
    "origin_station_name",
    "destination_station_name",
    "departure_date",
    "price",
    "departure_time",
    "group_size",
    "fare_type"],
  "Calendar_1": [
    "event_date",
    "event_time",
    "event_location",
    "event_name",
    "available_start_time",
    "available_end_time"],
  "Events_2": [
    "event_type",
    "category",
    "event_name",
    "date",
    "time",
    "number_of_tickets",
    "city",
    "venue",
    "venue_address"],
  "Flights_1": [
    "passengers",
    "seating_class",
    "origin_city",
    "destination_city",
    "origin_airport",
    "destination_airport",
    "departure_date",
    "return_date",
    "number_stops",
    "outbound_departure_time",
    "outbound_arrival_time",
    "inbound_arrival_time",
    "inbound_departure_time",
    "price",
    "refundable",
    "airlines"],
  "Flights_2": [
    "passengers",
    "seating_class",
    "origin",
    "destination",
    "origin_airport",
    "destination_airport",
    "departure_date",
    "return_date",
    "number_stops",
    "outbound_departure_time",
    "outbound_arrival_time",
    "inbound_arrival_time",
    "inbound_departure_time",
    "fare",
    "is_redeye",
    "airlines"],
  "Hotels_3": [
    "location",
    "number_of_rooms",
    "check_in_date",
    "check_out_date",
    "average_rating",
    "hotel_name",
    "street_address",
    "phone_number",
    "price",
    "pets_welcome"],
  "Media_1": [
    "title",
    "genre",
    "subtitles",
    "directed_by"],
  "Music_2": [
    "song_name",
    "artist",
    "album",
    "genre",
    "playback_device"],
  "RentalCars_2": [
    "car_type",
    "car_name",
    "pickup_location",
    "pickup_date",
    "pickup_time",
    "pickup_city",
    "dropoff_date",
    "total_price"],
  "Restaurants_1": [
    "restaurant_name",
    "date",
    "time",
    "serves_alcohol",
    "has_live_music",
    "phone_number",
    "street_address",
    "party_size",
    "price_range",
    "city",
    "cuisine"],
  "Services_2": [
    "dentist_name",
    "phone_number",
    "address",
    "city",
    "appointment_date",
    "appointment_time",
    "offers_cosmetic_services"],
  "Services_3": [
    "doctor_name",
    "phone_number",
    "average_rating",
    "street_address",
    "city",
    "appointment_date",
    "appointment_time",
    "type"
  ]
}

DOMAIN_SLOTS_DSTC = {
    "restaurant": ["food", "area", "name", "pricerange"]
}

DOMAIN_SLOTS_GSIM = {
  "movies": [
    "time",
    "num_tickets",
    "movie",
    "date",
    "theatre_name"
  ],
  "restaurant": [
    "num_people",
    "restaurant_name",
    "date",
    "time",
    "meal",
    "location",
    "price_range",
    "category",
    "rating"
  ]
}

DOMAIN_SLOTS_TT = {
    'movie': ["name.movie", "name.theater", "date.showing", "date.release", "description.plot", "description.other",
                "duration.movie", "location", "name.character", "name.genre", "name.person",
                "num.tickets", "price.ticket", "price.total", "rating.movie", "review.audience",
                "review.critic", "seating", "time.preference", "time.showing", "type.screening"]
}


DOMAIN_SLOTS_ABCD = {
  "product defect": ["order id", "order slotval", "customer name", "payment method", 
      "details slotval", "membership level", "amount", "username", "product", "refund target", "account id", 
      "company team", "reason slotval", "email", "zip code", "shipping option"], 
   "shipping issue": ["order id", "order slotval", "customer name", "change option", "payment method", 
      "full address", "membership level", "amount", "street address", "username", "product",
       "refund target", "account id", "reason slotval", "company team", "email", "zip code", "shipping option"], 
  "subscription inquiry": ["order id", "payment method", "customer name", "details slotval", "membership level",
      "product", "account slotval", "refund target", "account id", "company team",
       "amount", "zip code", "shipping option"], 
  "account access": ["order slotval", "details slotval", "customer name", "membership level", "amount",
      "shipping option", "pin number", "username", "company team", "email", "zip code", "phone"], 
  "troubleshoot site": ["order slotval", "details slotval", "customer name", "membership level", "amount",
      "username", "product", "account slotval", "company team", "email", "shipping option"], 
  "order issue": ["order id", "order slotval", "customer name", "change option", "payment method",
      "membership level", "product", "account slotval", "refund target", "account id", 
      "reason slotval", "company team", "amount", "zip code", "shipping option"], 
  "purchase dispute": ["order slotval", "order id", "customer name", "change option", "payment method",
      "membership level", "amount", "phone", "username", "product", "account slotval", 
      "account id", "company team", "reason slotval", "email", "zip code", "shipping option"], 
  "storewide query": ["payment method", "order slotval", "customer name", "change option",
      "membership level", "product", "account slotval", "company team", "reason slotval", 
      "zip code", "shipping option"], 
  "manage account": ["payment method", "order id", "customer name", "order slotval", "membership level",
      "shipping option", "street address", "username", "product", "account slotval", 
  "reason slotval", "company team", "amount", "zip code", "phone"], 
  "single item query": ["customer name", "membership level", "product", "company team", "amount",
      "shipping option"]
}

"""
<customer> i would like some soup and crackers.
<agent> what kind of soup do you want?
<customer> i want cream of broccoli.

speaker is "customer"
text is "i would like cream of broccoli."
utterance is "<customer> i would like cream of broccoli."
    utterance includes: speaker + text
    there are 3 utterances in this conversation
history is first two utterances
we do not use the term "context" anywhere
    context will always refer to support set examples
history + current_utt = dialogue
"""
