function restartService()
{
    setTimeout(location.reload, 3000);
    jQuery.get("/restart", function(data) {
    	console.log("setting reload...");
    })
    
}