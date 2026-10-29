Feature: `mc update` never reports a silently empty run (Story 10.2)
  A daily cron job that exits 0 and records a successful ingest while the
  archive stops growing is worse than a loud failure: nothing alerts. These
  guards make an empty discovery, an unprocessable backlog and a transient
  download failure all fail the run (exit 1) and leave `last_success`
  untouched, so the health gate keeps the staleness alert armed.

  Scenario: Discovery returns no videos
    Given channels return no video IDs
    When `mc update` runs
    Then the run exits 1
    And the run log reports an empty discovery
    And no successful ingest is recorded

  Scenario: Every unprocessed video is ignore-listed
    Given channels return only video "stuckvid001" and it is ignore-listed
    When `mc update` runs
    Then the run exits 1
    And the run log reports no progress
    And no successful ingest is recorded

  Scenario: A transient download failure is not ignore-listed
    Given channels return only video "newvid00001" and its download is rate limited
    When `mc update` runs
    Then the run exits 1
    And the run log reports a transient failure
    And video "newvid00001" is not ignore-listed
    And no successful ingest is recorded

  Scenario: A genuinely up-to-date archive succeeds
    Given channels return only video "oldvid00001" and it is already processed
    When `mc update` runs
    Then the run exits 0
    And the run log reports nothing to do
    And a successful ingest is recorded
