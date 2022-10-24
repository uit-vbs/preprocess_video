import meilisearch
import json
import glob

from tqdm import tqdm

api_key = "MASTER_KEY"
client = meilisearch.Client("http://192.168.20.156:7700", api_key)

# index_result_lst = []
# index_id = 0

# for json_file in sorted(tqdm(glob.glob('features/*.json'))):
#     with open(json_file, 'r') as f:
#         ocr_result_lst = json.load(f)
#         for result in ocr_result_lst['transition_ocr_result']:
#             result.update({"video_id": ocr_result_lst['video_id'][0]})
#             result.update({"id": index_id})
#             index_result_lst.append(result)
#             index_id += 1

documents = [
    {
        "id": 2,
        "title": "Ariel",
        "overview": "Taisto Kasurinen is a Finnish coal miner whose father has just committed suicide and who is framed for a crime he did not commit. In jail, he starts to dream about leaving the country and starting a new life. He escapes from prison but things don't go as planned...",
        "genres": ["Drama", "Crime", "Comedy"],
        "poster": "https://image.tmdb.org/t/p/w500/ojDg0PGvs6R9xYFodRct2kdI6wC.jpg",
        "release_date": 593395200,
    },
    {
        "id": 5,
        "title": "Four Rooms",
        "overview": "It's Ted the Bellhop's first night on the job...and the hotel's very unusual guests are about to place him in some outrageous predicaments. It seems that this evening's room service is serving up one unbelievable happening after another.",
        "genres": ["Crime", "Comedy"],
        "poster": "https://image.tmdb.org/t/p/w500/75aHn1NOYXh4M7L5shoeQ6NGykP.jpg",
        "release_date": 818467200,
    },
    {
        "id": 6,
        "title": "Judgment Night",
        "overview": "While racing to a boxing match, Frank, Mike, John and Rey get more than they bargained for. A wrong turn lands them directly in the path of Fallon, a vicious, wise-cracking drug lord. After accidentally witnessing Fallon murder a disloyal henchman, the four become his unwilling prey in a savage game of cat & mouse as they are mercilessly stalked through the urban jungle in this taut suspense drama",
        "genres": ["Action", "Thriller", "Crime"],
        "poster": "https://image.tmdb.org/t/p/w500/rYFAvSPlQUCebayLcxyK79yvtvV.jpg",
        "release_date": 750643200,
    },
    {
        "id": 11,
        "title": "Star Wars",
        "overview": "Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire. Venturesome Luke Skywalker and dashing captain Han Solo team together with the loveable robot duo R2-D2 and C-3PO to rescue the beautiful princess and restore peace and justice in the Empire.",
        "genres": ["Adventure", "Action", "Science Fiction"],
        "poster": "https://image.tmdb.org/t/p/w500/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg",
        "release_date": 233366400,
    },
    {
        "id": 12,
        "title": "Finding Nemo",
        "overview": "Nemo, an adventurous young clownfish, is unexpectedly taken from his Great Barrier Reef home to a dentist's office aquarium. It's up to his worrisome father Marlin and a friendly but forgetful fish Dory to bring Nemo home -- meeting vegetarian sharks, surfer dude turtles, hypnotic jellyfish, hungry seagulls, and more along the way.",
        "genres": ["Animation", "Family"],
        "poster": "https://image.tmdb.org/t/p/w500/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg",
        "release_date": 1054252800,
    },
    {
        "id": 13,
        "title": "Forrest Gump",
        "overview": "A man with a low IQ has accomplished great things in his life and been present during significant historic events—in each case, far exceeding what anyone imagined he could do. But despite all he has achieved, his one true love eludes him.",
        "genres": ["Comedy", "Drama", "Romance"],
        "poster": "https://image.tmdb.org/t/p/w500/h5J4W4veyxMXDMjeNxZI46TsHOb.jpg",
        "release_date": 773452800,
    },
    {
        "id": 14,
        "title": "American Beauty",
        "overview": "Lester Burnham, a depressed suburban father in a mid-life crisis, decides to turn his hectic life around after developing an infatuation with his daughter's attractive friend.",
        "genres": ["Drama"],
        "poster": "https://image.tmdb.org/t/p/w500/wby9315QzVKdW9BonAefg8jGTTb.jpg",
        "release_date": 937353600,
    },
    {
        "id": 15,
        "title": "Citizen Kane",
        "overview": "Newspaper magnate, Charles Foster Kane is taken from his mother as a boy and made the ward of a rich industrialist. As a result, every well-meaning, tyrannical or self-destructive move he makes for the rest of his life appears in some way to be a reaction to that deeply wounding event.",
        "genres": ["Mystery", "Drama"],
        "poster": "https://image.tmdb.org/t/p/w500/zO5OI25cieQ6ncpvGOD4U72vi1o.jpg",
        "release_date": -905990400,
    },
    {
        "id": 16,
        "title": "Dancer in the Dark",
        "overview": "Selma, a Czech immigrant on the verge of blindness, struggles to make ends meet for herself and her son, who has inherited the same genetic disorder and will suffer the same fate without an expensive operation. When life gets too difficult, Selma learns to cope through her love of musicals, escaping life's troubles - even if just for a moment - by dreaming up little numbers to the rhythmic beats of her surroundings.",
        "genres": ["Drama", "Crime"],
        "poster": "https://image.tmdb.org/t/p/w500/9rsivF4sWfmBzrNr4LPu6TNJhXX.jpg",
        "release_date": 958521600,
    },
    {
        "id": 17,
        "title": "The Dark",
        "overview": "Adèle and her daughter Sarah are traveling on the Welsh coastline to see her husband James when Sarah disappears. A different but similar looking girl appears who says she died in a past time. Adèle tries to discover what happened to her daughter as she is tormented by Celtic mythology from the past.",
        "genres": ["Horror", "Thriller", "Mystery"],
        "poster": "https://image.tmdb.org/t/p/w500/wZeBHVnCvaS2bwkb8jFQ0PwZwXq.jpg",
        "release_date": 1127865600,
    },
    {
        "id": 18,
        "title": "The Fifth Element",
        "overview": "In 2257, a taxi driver is unintentionally given the task of saving a young girl who is part of the key that will ensure the survival of humanity.",
        "genres": ["Adventure", "Fantasy", "Action", "Thriller", "Science Fiction"],
        "poster": "https://image.tmdb.org/t/p/w500/fPtlCO1yQtnoLHOwKtWz7db6RGU.jpg",
        "release_date": 862531200,
    },
    {
        "id": 19,
        "title": "Metropolis",
        "overview": "In a futuristic city sharply divided between the working class and the city planners, the son of the city's mastermind falls in love with a working class prophet who predicts the coming of a savior to mediate their differences.",
        "genres": ["Drama", "Science Fiction"],
        "poster": "https://image.tmdb.org/t/p/w500/hUK9rewffKGqtXynH5SW3v9hzcu.jpg",
        "release_date": -1353888000,
    },
    {
        "id": 20,
        "title": "My Life Without Me",
        "overview": "A fatally ill mother with only two months to live creates a list of things she wants to do before she dies without telling her family of her illness.",
        "genres": ["Drama", "Romance"],
        "poster": "https://image.tmdb.org/t/p/w500/sFSkn5rrQqXJkRNa2rMWqzmEuhR.jpg",
        "release_date": 1046995200,
    },
]

index = client.index("movies")
index.add_documents(documents)
