import os
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

EMBED_MODEL = "all-MiniLM-L6-v2"

STATIC_TOP_K = 5

DYNAMIC_TOP_K = 2

MAX_HOPS = 3


# ============================================================
# DOCUMENTS
# ============================================================

documents = [

    # --------------------------------------------------------
    # INCEPTION
    # --------------------------------------------------------

    {
        "id": "d1",
        "text": (
            "Dom Cobb is a professional extractor who steals "
            "secrets from the subconscious of sleeping people."
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
            "The dream-infiltration story involving Dom Cobb "
            "is titled Inception."
        )
    },

    {
        "id": "d4",
        "text": (
            "Inception was written and directed by "
            "Christopher Nolan."
        )
    },

    {
        "id": "d5",
        "text": (
            "Christopher Nolan later directed the World War II "
            "film Dunkirk."
        )
    },


    # --------------------------------------------------------
    # INTERSTELLAR
    # --------------------------------------------------------

    {
        "id": "d6",
        "text": (
            "Cooper is a former NASA pilot who becomes the "
            "central character in a mission through a wormhole."
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
            "The science-fiction film centered on Cooper's "
            "mission through a wormhole is Interstellar."
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
            "Ron Woodroof was a Texas electrician who organized "
            "an underground treatment network after being "
            "diagnosed with AIDS."
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
            "The biographical drama about Ron Woodroof is "
            "titled Dallas Buyers Club."
        )
    },

    {
        "id": "d13",
        "text": (
            "Dallas Buyers Club was directed by "
            "Jean-Marc Vallée."
        )
    },


    # --------------------------------------------------------
    # WILD
    # --------------------------------------------------------

    {
        "id": "d14",
        "text": (
            "Cheryl Strayed wrote a memoir about walking more "
            "than a thousand miles along the Pacific Crest Trail."
        )
    },

    {
        "id": "d15",
        "text": (
            "Reese Witherspoon portrayed Cheryl Strayed in the "
            "film adaptation of her memoir."
        )
    },

    {
        "id": "d16",
        "text": (
            "The film about Cheryl Strayed's long-distance "
            "hike is called Wild."
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
            "Elle Woods is a fashion-conscious student who "
            "attends Harvard Law School to win back her "
            "former boyfriend."
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
            "Bruce Wayne operates in Gotham City under the "
            "identity of a masked vigilante."
        )
    },

    {
        "id": "d23",
        "text": (
            "Christian Bale portrayed Bruce Wayne in "
            "The Dark Knight."
        )
    },

    {
        "id": "d24",
        "text": (
            "The film featuring the Joker's conflict with "
            "Gotham's masked vigilante is The Dark Knight."
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
            "Forrest Gump is a man from Alabama whose life "
            "intersects with several major events in American "
            "history."
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
            "The film following the life of Forrest Gump is "
            "titled Forrest Gump."
        )
    },

    {
        "id": "d29",
        "text": (
            "Forrest Gump was directed by Robert Zemeckis."
        )
    },


    # --------------------------------------------------------
    # THE MARTIAN
    # --------------------------------------------------------

    {
        "id": "d30",
        "text": (
            "Mark Watney is an astronaut stranded alone on "
            "Mars after his crew believes he has died."
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
            "The survival film about Mark Watney being stranded "
            "on Mars is The Martian."
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
            "The film centered on Tyler Durden and an "
            "underground fighting organization is Fight Club."
        )
    },

    {
        "id": "d37",
        "text": (
            "Fight Club was directed by David Fincher."
        )
    },


    # --------------------------------------------------------
    # DISTRACTORS
    # --------------------------------------------------------

    {
        "id": "d38",
        "text": (
            "Christopher Nolan's filmography includes films "
            "such as Memento, The Prestige, Inception, "
            "and Dunkirk."
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
            "Matthew McConaughey appeared in several dramas "
            "and science-fiction films during his career."
        )
    },

    {
        "id": "d41",
        "text": (
            "Reese Witherspoon has worked as both an actor "
            "and producer on numerous films."
        )
    },

    {
        "id": "d42",
        "text": (
            "Jean-Marc Vallée was a Canadian filmmaker known "
            "for character-driven dramas."
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
            "Ridley Scott has directed films spanning science "
            "fiction, historical drama, and crime."
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
# QUESTIONS
# ============================================================

questions = [

    {
        "question": (
            "Which director was responsible for the film "
            "centered on the professional extractor who "
            "steals secrets from sleeping people's "
            "subconscious?"
        ),
        "answer": "Christopher Nolan"
    },

    {
        "question": (
            "Which actor portrayed the astronaut whose "
            "mission involved traveling through a wormhole?"
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
            "Which actor portrayed the woman who walked more "
            "than a thousand miles along the Pacific Crest Trail?"
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
            "Who directed the film about an astronaut left "
            "alone on Mars after his crew believes he has died?"
        ),
        "answer": "Ridley Scott"
    },

    {
        "question": (
            "Which actor portrayed Tyler Durden in Fight Club?"
        ),
        "answer": "Brad Pitt"
    },
]


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBED_MODEL
)


# ============================================================
# GROQ CLIENT
# ============================================================

if "GROQ_API_KEY" not in os.environ:

    raise RuntimeError(
        "GROQ_API_KEY is not set.\n"
        "Run:\n"
        "export GROQ_API_KEY='your_key'"
    )


client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

print("Building FAISS index...")

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

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"Indexed {len(documents)} documents."
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k
):

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
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
# GROQ LLM
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

    usage = response.usage

    return {
        "text":
            response.choices[0].message.content.strip(),

        "input_tokens":
            usage.prompt_tokens,

        "output_tokens":
            usage.completion_tokens,

        "total_tokens":
            usage.total_tokens
    }


# ============================================================
# FINAL ANSWER
# ============================================================

def answer_question(
    question,
    evidence
):

    context = "\n".join(

        f"[{x['id']}] {x['text']}"

        for x in evidence
    )

    prompt = f"""
Answer the question using ONLY the evidence below.

Question:
{question}

Evidence:
{context}

Return only the final answer.
Do not explain your reasoning.
Do not invent information.
"""

    return llm(prompt)


# ============================================================
# DYNAMIC CONTROLLER
# ============================================================

def decide_next_step(
    question,
    evidence
):

    context = "\n".join(

        f"[{x['id']}] {x['text']}"

        for x in evidence
    )

    prompt = f"""
You are controlling an iterative retrieval system.

Question:
{question}

Evidence retrieved so far:
{context}

Decide whether the evidence is sufficient to answer the
question.

If the evidence is sufficient, respond exactly:

SUFFICIENT

If another retrieval step is required, respond in this format:

RETRIEVE: <short search query>

The new query should retrieve information that connects
the evidence already found to the final answer.

Do not answer the original question.
"""

    return llm(prompt)


# ============================================================
# STATIC RAG
# ============================================================

def static_rag(
    question
):

    evidence = retrieve(
        question,
        STATIC_TOP_K
    )

    result = answer_question(
        question,
        evidence
    )

    return {

        "answer":
            result["text"],

        "retrieval_calls":
            1,

        "chunks":
            len(evidence),

        "context_words":
            sum(
                len(x["text"].split())
                for x in evidence
            ),

        "input_tokens":
            result["input_tokens"],

        "output_tokens":
            result["output_tokens"],

        "total_tokens":
            result["total_tokens"],

        "hops":
            1,

        "evidence":
            evidence
    }


# ============================================================
# DYNAMIC RAG
# ============================================================

def dynamic_rag(
    question
):

    evidence = []

    seen_ids = set()

    query = question

    retrieval_calls = 0

    hop_count = 0

    total_input_tokens = 0

    total_output_tokens = 0

    total_tokens = 0


    for hop in range(
        MAX_HOPS
    ):

        hop_count += 1


        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        results = retrieve(
            query,
            DYNAMIC_TOP_K
        )

        retrieval_calls += 1


        # ----------------------------------------------------
        # ADD NEW EVIDENCE
        # ----------------------------------------------------

        for result in results:

            if result["id"] not in seen_ids:

                evidence.append(
                    result
                )

                seen_ids.add(
                    result["id"]
                )


        # ----------------------------------------------------
        # PRINT RETRIEVAL
        # ----------------------------------------------------

        print()

        print(
            f"      Hop {hop_count}"
        )

        print(
            f"      Query: {query}"
        )

        print(
            "      Retrieved:"
        )

        for result in results:

            print(
                f"        "
                f"{result['id']} "
                f"(score={result['score']:.3f})"
            )


        # ----------------------------------------------------
        # CONTROLLER
        # ----------------------------------------------------

        controller = decide_next_step(
            question,
            evidence
        )

        # total_input_tokens += (
        #     controller["input_tokens"]
        # )

        # total_output_tokens += (
        #     controller["output_tokens"]
        # )

        # total_tokens += (
        #     controller["total_tokens"]
        # )

        decision = controller["text"]


        print(
            f"      Controller: {decision}"
        )


        # ----------------------------------------------------
        # SUFFICIENT
        # ----------------------------------------------------

        if decision.strip().upper().startswith(
            "SUFFICIENT"
        ):

            break


        # ----------------------------------------------------
        # NEXT QUERY
        # ----------------------------------------------------

        match = re.search(
            r"RETRIEVE:\s*(.*)",
            decision,
            re.IGNORECASE
        )

        if not match:

            break


        query = match.group(1).strip()


        if not query:

            break


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    result = answer_question(
        question,
        evidence
    )

    total_input_tokens += (
        result["input_tokens"]
    )

    total_output_tokens += (
        result["output_tokens"]
    )

    total_tokens += (
        result["total_tokens"]
    )


    return {

        "answer":
            result["text"],

        "retrieval_calls":
            retrieval_calls,

        "chunks":
            len(evidence),

        "context_words":
            sum(
                len(x["text"].split())
                for x in evidence
            ),

        "input_tokens":
            total_input_tokens,

        "output_tokens":
            total_output_tokens,

        "total_tokens":
            total_tokens,

        "hops":
            hop_count,

        "evidence":
            evidence
    }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    text
):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9 ]",
        "",
        text
    )

    return " ".join(
        text.split()
    )


# ============================================================
# ACCURACY
# ============================================================

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

    return (
        expected in prediction
    )


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark():

    static_results = []

    dynamic_results = []


    print()

    print("=" * 75)

    print(
        "STATIC RAG vs DYNAMIC RAG"
    )

    print("=" * 75)

    print()

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"Static TOP_K: {STATIC_TOP_K}"
    )

    print(
        f"Dynamic TOP_K per hop: {DYNAMIC_TOP_K}"
    )

    print(
        f"Maximum dynamic hops: {MAX_HOPS}"
    )


    # ========================================================
    # QUESTIONS
    # ========================================================

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected = item["answer"]


        print()
        print()
        print("#" * 75)

        print(
            f"QUESTION {i}"
        )

        print("#" * 75)

        print()

        print(
            question
        )

        print(
            f"\nExpected: {expected}"
        )


        # ----------------------------------------------------
        # STATIC
        # ----------------------------------------------------

        static = static_rag(
            question
        )

        static["correct"] = (
            is_correct(
                static["answer"],
                expected
            )
        )


        # ----------------------------------------------------
        # DYNAMIC
        # ----------------------------------------------------

        print()

        print(
            "DYNAMIC RETRIEVAL"
        )

        print("-" * 60)


        dynamic = dynamic_rag(
            question
        )

        dynamic["correct"] = (
            is_correct(
                dynamic["answer"],
                expected
            )
        )


        static_results.append(
            static
        )

        dynamic_results.append(
            dynamic
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        print()

        print(
            "RESULT"
        )

        print("-" * 60)

        print(
            f"Static answer:  "
            f"{static['answer']}"
        )

        print(
            f"Static correct: "
            f"{static['correct']}"
        )

        print()

        print(
            f"Dynamic answer: "
            f"{dynamic['answer']}"
        )

        print(
            f"Dynamic correct:"
            f" {dynamic['correct']}"
        )

        print()

        print(
            f"Static:"
            f" {static['retrieval_calls']} retrieval,"
            f" {static['chunks']} chunks,"
            f" {static['context_words']} words,"
            f" {static['total_tokens']} total tokens"
        )

        print(
            f"Dynamic:"
            f" {dynamic['retrieval_calls']} retrieval,"
            f" {dynamic['chunks']} chunks,"
            f" {dynamic['context_words']} words,"
            f" {dynamic['total_tokens']} total tokens"
        )


    # ========================================================
    # AVERAGE
    # ========================================================

    def average(
        results,
        key
    ):

        return (

            sum(
                r[key]
                for r in results
            )

            /

            len(results)
        )


    static_accuracy = (

        sum(
            r["correct"]
            for r in static_results
        )

        /

        len(static_results)
    )


    dynamic_accuracy = (

        sum(
            r["correct"]
            for r in dynamic_results
        )

        /

        len(dynamic_results)
    )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print()

    print("=" * 75)

    print(
        "OVERALL RESULTS"
    )

    print("=" * 75)

    print()

    print(
        f"{'Metric':30s}"
        f"{'Static RAG':18s}"
        f"{'Dynamic RAG':18s}"
    )

    print(
        "-" * 75
    )


    print(
        f"{'Accuracy':30s}"
        f"{static_accuracy * 100:>8.1f}%"
        f"{'':10s}"
        f"{dynamic_accuracy * 100:>8.1f}%"
    )


    print(
        f"{'Avg retrieval calls':30s}"
        f"{average(static_results, 'retrieval_calls'):>8.2f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'retrieval_calls'):>8.2f}"
    )


    print(
        f"{'Avg evidence chunks':30s}"
        f"{average(static_results, 'chunks'):>8.2f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'chunks'):>8.2f}"
    )


    print(
        f"{'Avg context words':30s}"
        f"{average(static_results, 'context_words'):>8.1f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'context_words'):>8.1f}"
    )


    print(
        f"{'Avg input tokens':30s}"
        f"{average(static_results, 'input_tokens'):>8.1f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'input_tokens'):>8.1f}"
    )


    print(
        f"{'Avg output tokens':30s}"
        f"{average(static_results, 'output_tokens'):>8.1f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'output_tokens'):>8.1f}"
    )


    print(
        f"{'Avg total tokens':30s}"
        f"{average(static_results, 'total_tokens'):>8.1f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'total_tokens'):>8.1f}"
    )


    print(
        f"{'Avg hops':30s}"
        f"{average(static_results, 'hops'):>8.2f}"
        f"{'':10s}"
        f"{average(dynamic_results, 'hops'):>8.2f}"
    )


    print()

    print("=" * 75)

    print(
        "INTERPRETATION"
    )

    print("=" * 75)

    print()

    print(
        f"Static RAG retrieves {STATIC_TOP_K} documents "
        f"in a single round."
    )

    print(
        f"Dynamic RAG retrieves {DYNAMIC_TOP_K} documents "
        f"per round and can perform up to {MAX_HOPS} hops."
    )

    print()

    print(
        "Dynamic RAG performs additional retrieval only "
        "when the controller determines that the current "
        "evidence is insufficient."
    )

    print()

    print(
        "Token usage includes all controller calls and "
        "the final answer generation."
    )

    print()

    print(
        "This demo measures the accuracy-vs-retrieval-vs-token "
        "tradeoff for multi-hop questions."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_benchmark()