// ----- custom js ----- //

// hide initial
// $("#searching").hide();
// $("#results-table").hide();
$("#error").hide();

// global
var url = 'dataset/train';
var data = [];

$(function() {

  // sanity check
  console.log( "ready!" );

  // image click
  $(".img").click(function() {

    // empty/hide results
    $("#results").empty();
    // $("#results-table").hide();
    $("#error").hide();

    // add active class to clicked picture
    $(this).addClass("active")

    // grab image url
    var image = $(this).attr("src")
    var select = $('#selected_model').find(":selected").text();
    console.log(image)
    console.log(select)

    // show searching text
    $("#searching").show();
    console.log("searching...")

    // ajax request
    $.ajax({
      type: "POST",
      url: "/search",
      data : { img : image, select : select },
      // handle success
      success: function(result) {
        console.log(result.results);
        var data = result.results

        for (var i = 0; i < data.length; i++) {
          $("#results-table").append('<tr><th><p>'+data[i].image+'</p><img src="'+data[i].image+'" class="result-img"></th><th>'+data[i].score+'</th></tr>')
        };
      },
      // handle error
      error: function(error) {
        console.log(error);
        // append to dom
        $("#error").append()
      }
    });

    // show table
$("#results-table").show();

  });

});