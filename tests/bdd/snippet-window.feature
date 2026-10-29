Feature: Snippet context window (Story 4.10)
  A result must show enough surrounding transcript to understand the topic —
  not a single line. YouTube's newer captions are heavily punctuated (one
  short sentence per snippet under the 4.4 rule) while the older archive is
  one giant unpunctuated "sentence". The snippet is therefore a ~400-char
  window grown around the best-matching sentence and cut with ellipses,
  harmonizing both caption eras.

  Scenario: Punctuated chunk shows surrounding context
    Given a punctuated chunk with the query topic in the middle
    When I build its snippet for "ilha das américas"
    Then the snippet contains sentences around the best match
    And the snippet is cut with ellipses at the edges
    And the snippet stays within the context cap

  Scenario: Unpunctuated chunk keeps the query visible
    Given an unpunctuated chunk with the query deep inside
    When I build its snippet for "ilha das américas"
    Then the query terms are marked
    And the snippet is cut with ellipses at the edges
    And the snippet stays within the context cap
