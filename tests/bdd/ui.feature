Feature: Search page (HTML + htmx)
  A fast, simple search page in Portuguese with live results (Story 4.8).

  Scenario: Live search as I type
    Given I am on the search page
    When I type "bolha" in the search box
    Then results appear without a full page reload (htmx)
    And each result shows title, episode, date, channel and snippet

  Scenario: Empty results distinguish states
    When I search for something with no match
    Then the page shows "Muriel folheou o acervo inteiro" with the archive freshness line
    And the page does not look identical to a loading state

  Scenario: Freshness is visible
    Given the archive was last updated 2026-10-09
    When I open the search page
    Then the footer shows "acervo atualizado até 2026-10-09"

  Scenario: The page teaches agents how to use it
    Given I am on the search page
    Then the page shows how to point an agent at Muriel

  Scenario: Archive status is visible below everything
    Given the archive has 3556 catalogued videos totalling 197342 seconds
    And the last episode loaded is #971 from 2026-10-20
    When I open the search page
    Then the status block is the last content inside the footer
    And it shows "último episódio" "#971 · 2026-10-20"
    And it shows "vídeos catalogados" "3.556"
    And it shows "duração total" "197.342 segundos · 3.289,03 minutos · 54,82 horas · 2,28 dias"

  Scenario: Status hides when the archive reports nothing
    Given the archive has no recorded stats
    When I open the search page
    Then the page renders without the status block
    And the freshness line and colophon notes still show

  Scenario: A broken stats store never breaks the page
    Given the state store raises on stats reads
    When I open the search page
    Then the page still returns 200 with the search console and agent note intact

  Scenario: Agents read the same status
    Given the archive has 3556 catalogued videos totalling 197342 seconds
    When I request "/" with render=json
    Then the envelope includes "archive_status" with seconds 197342 and videos 3556

  Scenario: Sorting the current results by most recent
    Given the archive contains episodes dated "2026-10-20" and "2021-03-01" and an undated episode #1034
    When I search "bolha" with sort "recent"
    Then the first result shows date "2026-10-20"
    And the "sem data" result appears last
    And the caption reads "3 resultados · ordenado por mais recentes"

  Scenario: Invalid sort falls back to relevance on the human doors
    Given the archive contains episodes dated "2026-10-20" and "2021-03-01" and an undated episode #1034
    When I search "bolha" with sort "weird"
    Then the page renders 200 with relevance order

  Scenario: The sort control reflects and changes the active order
    Given I searched "bitcoin" with the default order
    Then the sort bar shows "mais relevantes" selected
    When I activate "mais antigos"
    Then the request URL carries "sort=oldest" and the page URL is "/?query=bitcoin&sort=oldest"

  Scenario: The active sort survives typing a new query
    Given I searched "bitcoin" with the default order
    Then the console input's htmx wiring includes the sort select

  Scenario: Static assets are cache-busted so a deploy never ships stale styles
    Given I am on the search page
    Then every vendored asset URL carries the file's content-hash version
    And /static responses demand revalidation with no-cache
