# Unit AI-1: Natural Language DB Interface

## Summary

You will be building an internal dashboard to query a SQL database using natural language. For obvious reasons, everyone at your company needs access to your curated collection of information about the Star Wars universe. 🌌

## Learning objectives

- Develop proficiency in producing desired LLM outputs through prompting.
- Acquire a deep understanding of prompt evaluation challenges and core strategies.
- Gain experience with creating a golden dataset to facilitate evaluation and iteration.
- Build a functional prototype of an LLM-driven full-stack application.

## Getting started

- Fork and clone this repo
- Add an `upstream` link to the CodesmithLLC repo
- Add a `partner` link to your partner's repo
- Use `git push origin main` and `git pull partner main` to stay in sync

## Challenges

### Database setup

- Verify that you can run the psql command: `psql --version`. If not:
  - Make sure postgresql is properly installed (`brew install postgresql` for Mac or [instructions here](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-database#install-postgresql) for WSL).
  - Add `psql` to your PATH (if necessary): Add the line `export PATH=$PATH:/Library/PostgreSQL/latest/bin` to your `~/.bashrc` or `~/.bash_profile`, respectively, and restart your terminal. The exact path may vary so be sure to confirm the location of the postgresql binaries.
- Create a new [Supabase](https://supabase.com/) project, set a password without special characters, and store the connection string in a `.env` file. Make sure to update [YOUR PASSWORD] to be the password you set.
- Seed your DB: `psql -d <Supabase connection string> -f starwars_postgres_create.sql`.

### App setup

- Create an OpenAI api key for this project (you will need to provide a credit card to make an account, but this unit should cost less than $1).
- Store your OPENAI_API_KEY (along with your SUPABASE_URI) in your `.env` file
- **TDD will be extremely beneficial**, especially as you get to the prompting portion! The less familiar you are with a given technology / challenge, the more helpful TDD can be. ✅
- Integrate your DB (you can create a `Pool` directly in the controller).
- Integrate `gpt-4o` for chat completions.

### Prompt evaluation

How can you be confident that your prompt works as expected? (This is the million dollar question!) The desired output here is relatively unambiguous insofar as we know what output the DB should produce for certain natural-language queries. But how can you test whether two SQL queries produce the same results (even if they're not identical)?

What are the best ways to evaluate your prompt? Which give you the most confidence? Which are most feasible? Which are fastest? Which are most cost efficient? These tradeoffs present very difficult engineering decisions, especially in production and at scale!

Some options to consider:

- Exact text match between generated SQL and ground truth SQL
- Levenshtein distance between SQL queries
- SQL parsing
- Comparing results of SQL queries executed on DB (note that you wouldn't typically test on your prod-db – but for simplicity let's go without a separate dev-db for testing)

### Prompt iteration

It’s a bit haphazard to just keep changing the prompt and hoping for the best. How can you be more methodical? How can you track your progress? What if a prompt works for one question but not another? Can you build a simple logging service (as middleware) that stores your prompt + input + output + any other relevant info?

Keep in mind the following goals:

- Accuracy (valid SQL queries that produce the intended results)
- Reliability (valid results at least X% of the time)
- Consistency (semantically equivalent queries treated similarly)
  - Flexibility (valid results with confusing / poorly worded input)
- Grounding (authoritative basis for assertions)
- Confidence (acknowledge uncertainty)
- Interpretability (be able to show reasoning / how response was generated)
- Alignment (not harmful, toxic, biased, dishonest, unreliable)
- Robustness (resist adversarial manipulation)

### A functional prototype!

You will have achieved sufficient functionality when you can successfully complete [Hunting the Hunter: A Star Wars Adventure](http://hunting-the-hunter1.us-west-2.elasticbeanstalk.com/) using your dashboard. You should enter information given in the challenge into the dashboard form – for example:

```text
story: 'The message is a short hologram showing a woman with black hair and brown eyes.'
question: 'What planet is she from?'
```

Fill in relevant context from previous answers as needed!

**Use strategies from the prompting lecture!**

And look at the API reference docs. In addition to modifying the prompt itself, are there other request body properties (like `temperature`) that you should specify and iterate on? Are there response body properties (like `logprobs`) that could give you additional insight?

### Build your golden (ground truth) dataset

To have confidence that your prompts are working as intended, you'll need to build a golden dataset to incorporate into your testing suite:

- Start with the Hunting the Hunter QA pairs.
- Expand it to build confidence across key criteria.
  - For example – you’ll need a variety of prompts to test consistency – maybe 5 similar ways of asking, 3 different, and 2 confusing.
  - Can you use an LLM to help with this – (how) should you?
- Include edge cases:
  - Complex queries (e.g. subqueries, complex joins, etc.)
  - Ambiguous inputs
- Are there public datasets you could use to augment your custom dataset?
- What size sample do you need in development? Pre-production? How do you weigh the time / API costs against the need for confidence in the model’s reliability?

### Further prompt evaluation

- You’ll want to be able to repeat your LLM tests to evaluate reliability. How many repetitions do you need to provide sufficient confidence? Can you make the number of repetitions customizable? (For example: you might want to repeat each test 10x as you iterate but 100x before you merge a change into production)
  - What pass rate is acceptable (i.e., what is your acceptance threshold)? How can you view this in a helpful way when you run the tests?
- Are there other metrics that would be helpful? What about perplexity (aka model confidence)? Are there benchmarks that would be helpful (at least to establish a reasonable acceptance threshold)?

## Extensions

Now that you’ve built a functional prototype and have adequate testing in place (for development purposes), it’s time to tackle other challenges to production deployment! You get to choose which of these to work on and in what order.

### How can you reduce costs?

- Can you use fewer tokens?
- Consider that gpt-4o-mini is (as of writing) 17x cheaper than gpt-4o! Switch to gpt-4o-mini and use your excellent testing setup to ensure your prompt still works well enough.

### How can you reduce latency?

- OpenAI has some [excellent suggestions](https://platform.openai.com/docs/guides/latency-optimization) for where to start here; in essence, you need to make fewer (consecutive) requests and/or make the LLM process requests more quickly. Also, make sure that you’re taking full advantage of OpenAI’s [prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
- (Alternatively / additionally you can focus on UX and perceived latency.)

### How can you improve the security of executing GPT-generated DB queries?

- Can you prevent SQL injection? Can you use parameterized queries or broader tools to escape dangerous characters?
- Can you restrict queries to predefined tables and columns?
- Can you limit the DB user’s permissions to be read-only?
- Can you validate the generated SQL using regular expressions and/or use a SQL parser to analyze the query structure?

### How can you monitor prompt performance?

- What's the best way to log inputs and outputs (and everything in between) in production? How do you version these with respect to your app (model version, prompts etc)?
- How should you solicit user feedback (and how should you track UA)?
- What metrics are most relevant? Which are most feasible to collect?

### How can you make your prompt extensible?

How adaptable is your prompt? What if your company decides to add a db for Star Trek? Or a TV show or something with a very different schema? How can you refine your application logic to facilitate further development / changing requirements? How can you modularize the prompt itself?

### How can you facilitate user-to-model interaction?

- How could you keep track of previous answers (in the same session) so the user can ask a series of questions without needing to provide previously "established" context?
- How should the model respond if it’s not sure what the user is asking? Some prompting strategies (like Cognitive Verifier) require further interaction between the user and the model. Could you facilitate dialog (when necessary) so the model can ask clarifying questions?
