Feature: Null-date payload backfill (Story 7.3)
  The temporal sort modes (mais recentes / mais antigos) are only as
  trustworthy as the payload `date` field. A newer episode can carry an
  episode number but a null payload date; the backfill heals it from the
  metadata cache so "mais recentes" ranks it by its true date instead of
  sinking it to the bottom.

  Scenario: Dateless new episode is healed and ranks by its true date
    Given the archive contains an episode dated "2026-10-20"
      And an undated episode #1034 whose payload date is null
      And the metadata cache reports the date "2026-10-22" for episode #1034
    When the dates backfill runs
    Then every indexed point for episode #1034 carries the date "2026-10-22"
    And a "recent" search ranks episode #1034 above the "2026-10-20" episode
