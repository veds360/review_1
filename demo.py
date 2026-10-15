# dynamic_rag_benchmark.py

import os
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-20b"

EMBED_MODEL = "all-MiniLM-L6-v2"

TOP_K = 2

MAX_HOPS = 3


# ============================================================
# KNOWLEDGE BASE
# ============================================================
# ============================================================
# HARDER MOVIE / PEOPLE KNOWLEDGE BASE
# ============================================================

documents = [

    # --------------------------------------------------------
    # INCEPTION CHAIN
    # --------------------------------------------------------

    {
        "id": "d1",
        "text": (
            "Dom Cobb is a professional extractor who steals secrets "
            "from the subconscious of sleeping people."
        )
    },

    {
        "id": "d2",
        "text": (
            "Leonardo DiCaprio portrayed the character Dom Cobb."
        )
    },

    {
        "id": "d3",
        "text": (
            "The dream-infiltration story involving Dom Cobb is titled "
            "Inception."
        )
    },

    {
        "id": "d4",
        "text": (
            "Inception was written and directed by Christopher Nolan."
        )
    },

    {
        "id": "d5",
        "text": (
            "Christopher Nolan later directed the World War II film "
            "Dunkirk."
        )
    },


    # --------------------------------------------------------
    # INTERSTELLAR CHAIN
    # --------------------------------------------------------

    {
        "id": "d6",
        "text": (
            "Cooper is a former NASA pilot who becomes the central "
            "character in a mission through a wormhole."
        )
    },

    {
        "id": "d7",
        "text": (
            "Matthew McConaughey portrayed the astronaut Cooper."
        )
    },

    {
        "id": "d8",
        "text": (
            "The science-fiction film centered on Cooper's mission "
            "through a wormhole is Interstellar."
        )
    },

    {
        "id": "d9",
        "text": (
            "Interstellar was directed by Christopher Nolan."
        )
    },


    # --------------------------------------------------------
    # DALLAS BUYERS CLUB
    # --------------------------------------------------------

    {
        "id": "d10",
        "text": (
            "Ron Woodroof was a Texas electrician who organized an "
            "underground treatment network after being diagnosed with AIDS."
        )
    },

    {
        "id": "d11",
        "text": (
            "Matthew McConaughey portrayed Ron Woodroof."
        )
    },

    {
        "id": "d12",
        "text": (
            "The biographical drama about Ron Woodroof is titled "
            "Dallas Buyers Club."
        )
    },

    {
        "id": "d13",
        "text": (
            "Dallas Buyers Club was directed by Jean-Marc Vallée."
        )
    },


    # --------------------------------------------------------
    # WILD
    # --------------------------------------------------------

    {
        "id": "d14",
        "text": (
            "Cheryl Strayed wrote a memoir about walking more than "
            "a thousand miles along the Pacific Crest Trail."
        )
    },

    {
        "id": "d15",
        "text": (
            "Reese Witherspoon portrayed Cheryl Strayed in the film "
            "adaptation of her memoir."
        )
    },

    {
        "id": "d16",
        "text": (
            "The film about Cheryl Strayed's long-distance hike is "
            "called Wild."
        )
    },

    {
        "id": "d17",
        "text": (
            "Wild was directed by Jean-Marc Vallée."
        )
    },


    # --------------------------------------------------------
    # LEGALLY BLONDE
    # --------------------------------------------------------

    {
        "id": "d18",
        "text": (
            "Elle Woods is a fashion-conscious student who attends "
            "Harvard Law School to win back her former boyfriend."
        )
    },

    {
        "id": "d19",
        "text": (
            "Reese Witherspoon played Elle Woods."
        )
    },

    {
        "id": "d20",
        "text": (
            "The comedy centered on Elle Woods is titled "
            "Legally Blonde."
        )
    },

    {
        "id": "d21",
        "text": (
            "Legally Blonde was directed by Robert Luketic."
        )
    },


    # --------------------------------------------------------
    # DARK KNIGHT
    # --------------------------------------------------------

    {
        "id": "d22",
        "text": (
            "Bruce Wayne operates in Gotham City under the identity "
            "of a masked vigilante."
        )
    },

    {
        "id": "d23",
        "text": (
            "Christian Bale portrayed Bruce Wayne in The Dark Knight."
        )
    },

    {
        "id": "d24",
        "text": (
            "The film featuring the Joker's conflict with Gotham's "
            "masked vigilante is The Dark Knight."
        )
    },

    {
        "id": "d25",
        "text": (
            "The Dark Knight was directed by Christopher Nolan."
        )
    },


    # --------------------------------------------------------
    # FORREST GUMP
    # --------------------------------------------------------

    {
        "id": "d26",
        "text": (
            "Forrest Gump is a man from Alabama whose life intersects "
            "with several major events in American history."
        )
    },

    {
        "id": "d27",
        "text": (
            "Tom Hanks portrayed Forrest Gump."
        )
    },

    {
        "id": "d28",
        "text": (
            "The film following the life of Forrest Gump is titled "
            "Forrest Gump."
        )
    },

    {
        "id": "d29",
        "text": (
            "Forrest Gump was directed by Robert Zemeckis."
        )
    },


    # --------------------------------------------------------
    # MARTIAN
    # --------------------------------------------------------

    {
        "id": "d30",
        "text": (
            "Mark Watney is an astronaut stranded alone on Mars "
            "after his crew believes he has died."
        )
    },

    {
        "id": "d31",
        "text": (
            "Matt Damon portrayed the astronaut Mark Watney."
        )
    },

    {
        "id": "d32",
        "text": (
            "The survival film about Mark Watney being stranded on "
            "Mars is The Martian."
        )
    },

    {
        "id": "d33",
        "text": (
            "The Martian was directed by Ridley Scott."
        )
    },


    # --------------------------------------------------------
    # FIGHT CLUB
    # --------------------------------------------------------

    {
        "id": "d34",
        "text": (
            "An unnamed office worker suffering from insomnia "
            "becomes involved with Tyler Durden."
        )
    },

    {
        "id": "d35",
        "text": (
            "Brad Pitt portrayed Tyler Durden."
        )
    },

    {
        "id": "d36",
        "text": (
            "The film centered on Tyler Durden and an underground "
            "fighting organization is Fight Club."
        )
    },

    {
        "id": "d37",
        "text": (
            "Fight Club was directed by David Fincher."
        )
    },


    # --------------------------------------------------------
    # EXTRA DISTRACTORS
    # --------------------------------------------------------

    {
        "id": "d38",
        "text": (
            "Christopher Nolan's filmography includes films such as "
            "Memento, The Prestige, Inception, and Dunkirk."
        )
    },

    {
        "id": "d39",
        "text": (
            "Leonardo DiCaprio has appeared in films including "
            "Titanic, The Revenant, and The Wolf of Wall Street."
        )
    },

    {
        "id": "d40",
        "text": (
            "Matthew McConaughey appeared in several dramas and "
            "science-fiction films during his career."
        )
    },

    {
        "id": "d41",
        "text": (
            "Reese Witherspoon has worked as both an actor and producer "
            "on numerous films."
        )
    },

    {
        "id": "d42",
        "text": (
            "Jean-Marc Vallée was a Canadian filmmaker known for "
            "character-driven dramas."
        )
    },

    {
        "id": "d43",
        "text": (
            "Robert Zemeckis directed several films involving "
            "historical or fantastical stories."
        )
    },

    {
        "id": "d44",
        "text": (
            "Ridley Scott has directed films spanning science fiction, "
            "historical drama, and crime."
        )
    },

    {
        "id": "d45",
        "text": (
            "David Fincher is known for psychological thrillers "
            "and dark dramas."
        )
    },
]


# ============================================================
# MULTI-HOP QUESTIONS
# ============================================================
# ============================================================
# HARDER QUESTIONS
# ============================================================

questions = [

    {
        "question": (
            "Which director was responsible for the film centered "
            "on the professional extractor who steals secrets "
            "from sleeping people's subconscious?"
        ),
        "answer": "Christopher Nolan"
    },

    {
        "question": (
            "Which actor portrayed the astronaut whose mission "
            "involved traveling through a wormhole?"
        ),
        "answer": "Matthew McConaughey"
    },

    {
        "question": (
            "Who directed the film about the Texas electrician "
            "who organized an underground treatment network?"
        ),
        "answer": "Jean-Marc Vallée"
    },

    {
        "question": (
            "Which actor portrayed the woman who walked more than "
            "a thousand miles along the Pacific Crest Trail?"
        ),
        "answer": "Reese Witherspoon"
    },

    {
        "question": (
            "Who directed the comedy about a fashion-conscious "
            "student who attends Harvard Law School?"
        ),
        "answer": "Robert Luketic"
    },

    {
        "question": (
            "Which actor portrayed the masked vigilante whose "
            "conflict with the Joker takes place in Gotham City?"
        ),
        "answer": "Christian Bale"
    },

    {
        "question": (
            "Who directed the film about an astronaut left alone "
            "on Mars after his crew believes he has died?"
        ),
        "answer": "Ridley Scott"
    },

    {
        "question": (
            "Which actor portrayed the character whose relationship "
            "with Tyler Durden forms the basis of Fight Club?"
        ),
        "answer": "Brad Pitt"
    },

]


# ============================================================
# MODELS
# ============================================================

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBED_MODEL
)


if "GROQ_API_KEY" not in os.environ:

    raise RuntimeError(
        "Set GROQ_API_KEY first."
    )


client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


# ============================================================
# BUILD FAISS
# ============================================================

texts = [
    d["text"]
    for d in documents
]

embeddings = embedder.encode(
    texts,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(embeddings)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query):

    vector = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    vector = np.asarray(
        vector,
        dtype="float32"
    )

    scores, indices = index.search(
        vector,
        TOP_K
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append({

            "id":
                documents[idx]["id"],

            "text":
                documents[idx]["text"],

            "score":
                float(score)

        })

    return results


# ============================================================
# LLM
# ============================================================

def llm(prompt):

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=250
    )

    return response.choices[0].message.content.strip()


# ============================================================
# APPROX TOKEN COUNT
# ============================================================

def token_count(text):

    # Approximation only.
    return len(text.split())


# ============================================================
# DYNAMIC CONTROLLER
# ============================================================

def assess(question, evidence):

    context = "\n".join(
        f"[{x['id']}] {x['text']}"
        for x in evidence
    )

    prompt = f"""
You are controlling an iterative retrieval system.

Original question:
{question}

Retrieved evidence:
{context}

Determine whether the evidence is sufficient to answer
the original question.

If it is not sufficient, generate ONE concise query that
would retrieve the missing information.

Return exactly:

SUFFICIENT: YES
NEXT_QUERY: NONE

or:

SUFFICIENT: NO
NEXT_QUERY: <query>

Do not answer the original question.
Do not provide chain-of-thought.
"""

    return llm(prompt)


# ============================================================
# PARSE CONTROLLER
# ============================================================

def parse_controller(output):

    sufficient = (
        "SUFFICIENT: YES"
        in output.upper()
    )

    next_query = None

    for line in output.splitlines():

        if line.upper().startswith(
            "NEXT_QUERY:"
        ):

            next_query = line.split(
                ":",
                1
            )[1].strip()

    return sufficient, next_query


# ============================================================
# FINAL ANSWER
# ============================================================

def answer_question(
    question,
    evidence
):

    context = "\n".join(
        x["text"]
        for x in evidence
    )

    prompt = f"""
Answer the question using only the evidence.

Question:
{question}

Evidence:
{context}

Return only the answer, preferably a short
name or phrase.
"""

    return llm(prompt)


# ============================================================
# NORMALIZE ANSWER
# ============================================================

def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9 ]",
        "",
        text
    )

    return " ".join(
        text.split()
    )


def is_correct(
    prediction,
    expected
):

    prediction = normalize(
        prediction
    )

    expected = normalize(
        expected
    )

    return expected in prediction


# ============================================================
# STATIC RAG
# ============================================================

def static_rag(question):

    evidence = retrieve(
        question
    )

    context = "\n".join(
        x["text"]
        for x in evidence
    )

    answer = answer_question(
        question,
        evidence
    )

    return {

        "answer": answer,

        "correct": None,

        "retrieval_calls": 1,

        "chunks": len(evidence),

        "context_tokens":
            token_count(context),

        "llm_calls": 1

    }


# ============================================================
# DYNAMIC RAG
# ============================================================

def dynamic_rag(question):

    evidence = []

    retrieval_calls = 0

    llm_calls = 0


    for hop in range(MAX_HOPS):

        # ----------------------------------------------
        # RETRIEVE
        # ----------------------------------------------

        query = (
            question
            if hop == 0
            else next_query
        )

        results = retrieve(
            query
        )

        retrieval_calls += 1


        # ----------------------------------------------
        # ADD NEW EVIDENCE
        # ----------------------------------------------

        existing = {
            x["id"]
            for x in evidence
        }

        for result in results:

            if result["id"] not in existing:

                evidence.append(
                    result
                )


        # ----------------------------------------------
        # ASSESS
        # ----------------------------------------------

        assessment = assess(
            question,
            evidence
        )

        llm_calls += 1

        sufficient, next_query = parse_controller(
            assessment
        )


        print(
            f"      hop {hop + 1}: "
            f"{[x['id'] for x in results]}"
        )


        if sufficient:

            break


        if not next_query:

            break


    # ----------------------------------------------
    # FINAL ANSWER
    # ----------------------------------------------

    answer = answer_question(
        question,
        evidence
    )

    llm_calls += 1


    context = "\n".join(
        x["text"]
        for x in evidence
    )


    return {

        "answer": answer,

        "correct": None,

        "retrieval_calls":
            retrieval_calls,

        "chunks":
            len(evidence),

        "context_tokens":
            token_count(context),

        "llm_calls":
            llm_calls

    }


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark():

    static_results = []

    dynamic_results = []


    print()
    print("=" * 75)
    print("MULTI-HOP RAG BENCHMARK")
    print("=" * 75)

    print(
        f"\nSame TOP_K for both systems: {TOP_K}"
    )


    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected = item["answer"]


        print()
        print("-" * 75)

        print(
            f"Question {i}:"
        )

        print(question)

        print(
            f"Expected: {expected}"
        )


        # ====================================================
        # STATIC
        # ====================================================

        static = static_rag(
            question
        )

        static["correct"] = is_correct(
            static["answer"],
            expected
        )


        # ====================================================
        # DYNAMIC
        # ====================================================

        print("\nDynamic retrieval:")

        dynamic = dynamic_rag(
            question
        )

        dynamic["correct"] = is_correct(
            dynamic["answer"],
            expected
        )


        static_results.append(
            static
        )

        dynamic_results.append(
            dynamic
        )


        print(
            "\nStatic:"
        )

        print(
            f"  Answer: {static['answer']}"
        )

        print(
            f"  Correct: {static['correct']}"
        )


        print(
            "\nDynamic:"
        )

        print(
            f"  Answer: {dynamic['answer']}"
        )

        print(
            f"  Correct: {dynamic['correct']}"
        )


    # ========================================================
    # AGGREGATE
    # ========================================================

    def avg(results, key):

        return sum(
            r[key]
            for r in results
        ) / len(results)


    static_accuracy = sum(
        r["correct"]
        for r in static_results
    ) / len(static_results)


    dynamic_accuracy = sum(
        r["correct"]
        for r in dynamic_results
    ) / len(dynamic_results)


    print()
    print()
    print("=" * 75)
    print("OVERALL RESULTS")
    print("=" * 75)

    print()

    print(
        f"{'Metric':30s}"
        f"{'Static RAG':18s}"
        f"{'Dynamic RAG':18s}"
    )

    print("-" * 70)

    print(
        f"{'Accuracy':30s}"
        f"{static_accuracy * 100:>6.1f}%"
        f"{'':12s}"
        f"{dynamic_accuracy * 100:>6.1f}%"
    )

    print(
        f"{'Avg retrieval calls':30s}"
        f"{avg(static_results, 'retrieval_calls'):>8.2f}"
        f"{'':10s}"
        f"{avg(dynamic_results, 'retrieval_calls'):>8.2f}"
    )

    print(
        f"{'Avg evidence chunks':30s}"
        f"{avg(static_results, 'chunks'):>8.2f}"
        f"{'':10s}"
        f"{avg(dynamic_results, 'chunks'):>8.2f}"
    )

    print(
        f"{'Avg context words':30s}"
        f"{avg(static_results, 'context_tokens'):>8.1f}"
        f"{'':10s}"
        f"{avg(dynamic_results, 'context_tokens'):>8.1f}"
    )

    print(
        f"{'Avg LLM calls':30s}"
        f"{avg(static_results, 'llm_calls'):>8.2f}"
        f"{'':10s}"
        f"{avg(dynamic_results, 'llm_calls'):>8.2f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_benchmark()