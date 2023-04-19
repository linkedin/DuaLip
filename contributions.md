# Contributions

## Agreement
As a contributor, you represent that the code you submit is your
original work or that of your employer (in which case you represent
you have the right to bind your employer).  By submitting code, you
(and, if applicable, your employer) are licensing the submitted code
to LinkedIn and the open source community subject to the BSD 2-Clause
license.

## Mailing Lists
If you want to get in touch regarding this project or have successful 
of applications of DuaLip, we would love to hear from you. Please send us an email
at [ask_lp@linkedin.com](mailto:ask_lp@linkedin.com).

## Responsible Disclosure of Security Vulnerabilities

**Do not file an issue on Github for security issues.**  Please review
the [guidelines for disclosure](https://www.linkedin.com/help/linkedin/answer/62924).  Reports should
be encrypted using PGP [public key](https://www.linkedin.com/help/linkedin/answer/79676)
and sent to [security@linkedin.com](mailto:security@linkedin.com?subject=Vulnerability%20in%20Github%20LinkedIn/Dualip%20-%20%3Csummary%3E)
preferably with the title "Vulnerability in Github LinkedIn/DuaLip - < short summary >".

## Steps for Creating Pull Request
1. Create a new local branch (git branch mybranch)
2. Make code changes in local branch
3. **Change major and/or minor version in `version.properties` if necessary**.
   We follow semantic versioning practices (major.minor.patch).
   * Increment Major for backward incompatible change.
   * Increment Minor for major function changes.
   * Patch versionining and releases are handled automatically by CI/CD flows.
4. Commit changes to local branch (git commit)
5. Test build (./gradlew build)
6. Push local changes to remote branch. (git push -f origin mybranch)
7. Create a new pull request at https://github.com/linkedin/DuaLip/pulls
8. If you need to update the pull request, repeat steps 2 through 6, **excluding step 3**.
9. Getting review and approval. Squash and merge to main branch.

## Tips for Getting Your Pull Request Accepted

1. Make sure all new features are tested and the tests pass.
2. Bug fixes must include a test case demonstrating the error that it fixes.
3. Open an issue first and seek advice for your change before submitting a pull request. Large features which have never been discussed are unlikely to be accepted.
