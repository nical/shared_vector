# Contributing

This project welcomes contribution from everyone. Here are the guidelines if you are
thinking of helping us:

## Contributions

Contributions to this project should be made in the form of GitHub pull requests.
Each pull request will be reviewed by a core contributor (someone with
permission to land patches) and either landed in the main tree or
given feedback for changes that would be required.
All contributions should follow this format, even those from core contributors.

Should you wish to work on an issue, please claim it first by commenting on
the GitHub issue that you want to work on it. This is to prevent duplicated
efforts from contributors on the same issue.

## Pull Request Checklist

- Branch from the main branch and, if needed, rebase to the current main
  branch before submitting your pull request. If it doesn't merge cleanly with
  main you may be asked to rebase your changes.

- Commits should be as small as possible, while ensuring that each commit is
  correct independently (i.e., each commit should compile and pass tests).

- If your patch is not getting reviewed or you need a specific person to review
  it, you can @-reply a reviewer asking for a review in the pull request or a
  comment.

- Whenever applicable, add tests relevant to the fixed bug or new feature.

For specific git instructions, see [GitHub workflow 101](https://github.com/servo/servo/wiki/Github-workflow).

## Testing

To run all tests, execute `cargo test` as well as `cargo +nightly miri test` from the root of the repository.

## Conduct

In all related forums, we follow the [Rust Code of Conduct](http://www.rust-lang.org/conduct.html).
For escalation or moderation issues, please contact [Nical](https://github.com/nical) instead of the Rust moderation team.

## License

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed dual MIT/Apache 2,without any additional terms or conditions.
