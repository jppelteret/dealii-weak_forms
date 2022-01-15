# Contributing
This library was developed solely as hobby for the author, who has produced this
work for their own personal interest and enjoyment. Although it is added to a
public repository, the author does not necessarily have the time nor the inclination
to develop the work in a collaborative manner. This is mainly due to time constraints
more than anything else (the author is already a principal developer for the
deal.II library, which occupies a significant portion of their limited free time).
Should time have not been an issue then the documentation for the project, as well
as examples demonstrating its capabilities, would be much more comprehensive.

That said, the author would like to think that this project can be extended to
meet the needs of users. So bugs, issues, and other comments can be reported as an
[issue](https://github.com/jppelteret/dealii-weak_forms/issues).
Depending on their severity and ease of resolution, they may be attended to
(but unfortunately there are no guarantees to that effect).
Feature requests can also be made in the form of a
[pull request](https://github.com/jppelteret/dealii-weak_forms/pulls)
although, with apologies, no guarantee can be made that they'd be accepted
(again, this is mainly due to time constraints). 


# Code Formatting
Code formatting is to be performed using `clang-format`, specifically version
`11` for which the `.clang-format` settings are optimsed.


# Tests
This library has been developed using a "test-driven" approach.
Therefore, tests covering each new feature are pretty much mandatory. 
This helps ensure that the abstractions that the library provides are accurate
and reliable -- something that the users of such a library demand.
Tests are performed using `CTest`, and can be added to the `tests` folder.
