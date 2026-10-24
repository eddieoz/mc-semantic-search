Feature: YouTube Data API fact backfill (Story 9.5)
  The temporal sort is only as truthful as the payload `date` field. The
  API backfill hydrates exact upload dates for the whole archive — without
  cookies, without scraping, without the bot gate — so "mais recentes"
  orders real dates.

  Scenario: Undated episode is dated by the API backfill and ranks by its true date
    Given the archive contains an episode dated "2026-10-20"
      And an undated episode #1035 whose payload date is null
      And the YouTube API reports upload date "2026-10-22" and duration 5400 for episode #1035's video
    When the API backfill runs
    Then the metadata cache for episode #1035's video carries the date "2026-10-22"
    And every indexed point for episode #1035 carries the date "2026-10-22"
    And a "recent" search ranks episode #1035 above the "2026-10-20" episode

  Scenario: A video YouTube no longer serves is counted, never fatal
    Given the archive contains an undated video the API does not return
      And another undated video the API does return
    When the API backfill runs
    Then the run completes with the unserved video counted as still missing
    And every other undated video is dated
